# Compute Engine deployment

Target: Ubuntu 24.04, e2-medium (4 GiB), 30 GiB balanced persistent disk.
Keep the VM near kwonserver and PostgreSQL. This is a single-node MVP deployment,
not a highly available installation. The API is on port 8001; Redis has no host port.

Install Docker Engine and the Compose plugin using Docker's official Ubuntu guide:
https://docs.docker.com/engine/install/ubuntu/
Enable Docker on boot with `sudo systemctl enable --now docker`.
Run the commands below from the kwonrec checkout using a Docker-authorized account.

## Configuration

Create `.env.production` on the VM with permissions 600. Do not commit it or
paste secrets into chat. Compose reads this file (including quoted dotenv values).

```dotenv
KWONREC_API_KEY=<same random key as kwonserver, at least 32 characters>
KWONREC_DATABASE_URL=<direct PostgreSQL connection URL with SSL enabled>
KWONREC_NAMESPACE=kwonrec:v2
KWONREC_BIND_IP=127.0.0.1
KWONREC_POLL_SECONDS=1
```

The one-second polling interval reduces idle database queries versus local
development's 250 ms interval, at the cost of additional ingestion latency.
The worker keeps PostgreSQL active: account for this in the database provider bill.
The setup role needs permission to create the outbox schema and triggers.
Use the least-privileged worker role described in README.MD after setup.

## First installation / moving from local Redis

Stop all existing workers consuming this database's kwonrec outbox before this
sequence. They must not split events across independent Redis instances. Keep
kwonserver on its chronological fallback during cutover. Back up existing state.

```sh
chmod 600 .env.production
dc() { docker compose --env-file .env.production -f deploy/compute-engine.compose.yml "$@"; }
dc config --quiet
dc build
dc up -d --wait redis
dc run --rm setup
dc run --rm setup python -m src.runtime.worker --replay-retained
dc up -d --wait api worker
curl --fail http://127.0.0.1:8001/ready
dc ps
```

Setup installs transactional capture and bootstraps posts/history. Replay restores
retained interactions already marked processed by the old worker. It cannot
recover expired/pruned history. Do not start the old worker after cutover.
API readiness only checks Redis, not ingestion completeness. Check worker logs,
outbox backlog/dead letters and an authenticated recommendation before switching
kwonserver. Never print `dc config` without `--quiet`: it contains secrets.

## Cloud Run connection

Configure kwonserver with Direct VPC egress into the VM's VPC/subnet (use a
compatible same-region subnet). Permit TCP 8001 only from the Cloud Run egress
subnet using a firewall rule targeted at this VM. Do not expose Redis or port
8001 to the public internet. Avoid an extra VPC connector just for this setup.

Set `KWONREC_BIND_IP` to the VM's reserved internal IPv4 address, then recreate
the API with `dc up -d api`. Set Cloud Run's `KWONREC_API` to
`http://<internal-ip>:8001` and `KWONREC_API_KEY` to the matching secret. The
environment variable is `KWONREC_API`, not `KWONREC_API_URL`.
Verify the feed reports `X-Feed-Source: kwonrec` with a logged-in test account.
Keep public frontend calls going through kwonserver.

Direct VPC setup: https://cloud.google.com/run/docs/configuring/vpc-direct-vpc

## Subsequent updates

```sh
dc build
dc up -d --wait api worker
dc ps
```

Apply future SQL migrations explicitly before dependent application updates.
Do not rerun bootstrap/replay on every code deployment. Tag images using
`KWONREC_IMAGE_TAG` for identifiable releases; keep the prior image for rollback.
This basic Compose update can briefly interrupt the API; kwonserver falls back.

## Operations

- Keep `redis_data` across deployments. Never use `docker compose down -v`.
- Configure persistent disk snapshots and test Redis recovery; AOF is not a backup.
- Monitor host disk/memory, Redis memory/write failures, worker restarts, outbox
  age/dead letters and feed fallback rate. 512 MiB Redis is a starting limit.
- Docker restarts exited processes and starts containers after reboot; an
  unhealthy-but-running process needs investigation (health checks do not restart it).
- Logs rotate automatically. Run `dc logs --tail 100 worker` for ingestion status;
  treat logs as private because connection errors may include infrastructure details.
- Set billing alerts; alerts do not cap charges. OS patching remains required.
