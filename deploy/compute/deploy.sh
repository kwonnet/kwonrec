#!/usr/bin/env bash
# Invoked automatically by GitHub Actions on the target Debian/Ubuntu VM.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run with sudo.' >&2; exit 1; }
IMAGE=${1:?Immutable image required}
[[ "$IMAGE" =~ ^[a-z0-9-]+-docker.pkg.dev/[a-z0-9-]+/kwonnet/kwonrec@sha256:[a-f0-9]{64}$ ]] || { echo 'Unexpected image reference' >&2; exit 2; }
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/opt/kwonrec
mkdir -p "$ROOT/env" "$ROOT/history"
chmod 700 "$ROOT" "$ROOT/env" "$ROOT/history"
exec 9>"$ROOT/.deploy.lock"
flock -w 1800 9
if ! command -v docker >/dev/null; then
  source /etc/os-release
  case "$ID" in debian|ubuntu) ;; *) echo 'Automatic Docker installation supports Debian/Ubuntu only.' >&2; exit 1 ;; esac
  apt-get update
  apt-get install -y ca-certificates curl python3
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/%s %s stable\n' "$(dpkg --print-architecture)" "$ID" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi
command -v python3 >/dev/null || { apt-get update; apt-get install -y python3; }
systemctl enable --now docker
WORK=$(mktemp -d "$ROOT/.release.XXXXXX")
trap 'rm -rf "$WORK"; rm -f "$BUNDLE/registry-token" "$BUNDLE/app.env"' EXIT
# Existing kwonserver installations may have Docker without the Compose plugin.
if ! docker compose version >/dev/null 2>&1; then
  apt-get update
  apt-get install -y docker-compose-plugin
fi
test -s "$BUNDLE/app.env" || { echo 'KWONREC_ENV must contain runtime KEY=value lines.' >&2; exit 1; }
install -m 600 "$BUNDLE/app.env" "$WORK/app.env"
python3 "$BUNDLE/validate-env.py" "$WORK/app.env" "$WORK"
python3 "$BUNDLE/check-cloud-logging.py"
export DOCKER_CONFIG="$WORK/docker"
mkdir -m 700 "$DOCKER_CONFIG"
docker login "${IMAGE%%/*}" --username oauth2accesstoken --password-stdin < "$BUNDLE/registry-token" >/dev/null
rm -f "$BUNDLE/registry-token"
install -m 600 "$BUNDLE/compose.yml" "$WORK/compose.yml"
printf 'DEPLOY_IMAGE=%s\nKWONREC_BIND_IP=%s\n' "$IMAGE" "$(cat "$WORK/binding")" > "$WORK/release.env"
dc() { docker compose --env-file "$WORK/release.env" -f "$WORK/compose.yml" "$@"; }
dc config --quiet
docker network inspect kwonserver-production >/dev/null 2>&1 || docker network create kwonserver-production >/dev/null
dc pull api worker redis
# Validate settings in the actual image, without contacting production or printing values.
dc run --rm --no-deps setup python -c 'from src.runtime.settings import settings; settings()' >/dev/null
dc up -d --wait --wait-timeout 90 redis
# Keep the existing Compose project and volume. Never run "down --volumes".
restore_previous() {
  echo 'Deployment failed; restoring the previous application release when available.' >&2
  python3 "$BUNDLE/diagnostics.py" "$WORK/app.env" "$WORK/setup.log" >&2 || true
  if [[ -f "$ROOT/current/compose.yml" ]]; then
    docker compose --env-file "$ROOT/current/release.env" -f "$ROOT/current/compose.yml" up -d api worker || true
  else
    # On adoption, the original worker container still exists if setup failed.
    dc start worker || true
  fi
}
trap 'restore_previous' ERR
dc stop worker
# Install/upgrade triggers each time; backfill only for a new database/Redis namespace.
if [[ -f "$ROOT/setup-target" && "$(cat "$ROOT/setup-target")" == "$(cat "$WORK/target")" ]]; then
  dc run --rm setup python -m src.runtime.setup --skip-bootstrap > "$WORK/setup.log" 2>&1
else
  dc run --rm setup > "$WORK/setup.log" 2>&1
  dc run --rm setup python -m src.runtime.worker --replay-retained >> "$WORK/setup.log" 2>&1
fi
# A real outbox batch checks database/Redis integration; worker process liveness alone is insufficient.
dc run --rm setup python -m src.runtime.worker --once >> "$WORK/setup.log" 2>&1
dc up -d --wait --wait-timeout 180 api worker
API=$(dc ps -q api)
WORKER=$(dc ps -q worker)
docker exec "$API" python -c 'import os,urllib.request; r=urllib.request.Request("http://localhost:8001/recommend/deployment-probe?limit=1",headers={"Authorization":"Bearer "+os.environ["KWONREC_API_KEY"]}); urllib.request.urlopen(r,timeout=10).read()'
sleep 5
[[ "$(docker inspect --format '{{.State.Running}} {{.RestartCount}}' "$WORKER")" == 'true 0' ]]
# Persist a reproducible release, with runtime secrets readable by root only.
if [[ -d "$ROOT/current" ]]; then mv "$ROOT/current" "$ROOT/history/$(date -u +%Y%m%dT%H%M%S)"; fi
mkdir -m 700 "$ROOT/current"
for file in app.env compose.yml release.env; do install -m 600 "$WORK/$file" "$ROOT/current/$file"; done
install -m 600 "$WORK/target" "$ROOT/setup-target"
trap - ERR
echo 'Kwonrec API and worker deployed. Same-VM kwonserver URL: http://kwonrec:8001'
