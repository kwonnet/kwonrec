"""Validate Docker env-file syntax without emitting values or secrets."""
import hashlib
import ipaddress
import pathlib
import re
import sys
from urllib.parse import urlsplit


def validate(text):
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise ValueError("Use unique KEY=value lines without export or spaces around keys")
        if value.startswith(('"', "'")):
            raise ValueError("Remove surrounding quotes from runtime values")
        values[key] = value
    if len(values.get("KWONREC_API_KEY", "")) < 32:
        raise ValueError("KWONREC_API_KEY must have at least 32 characters")
    for name, schemes in [("KWONREC_DATABASE_URL", ("postgres", "postgresql")),
                          ("KWONREC_REDIS_URL", ("redis", "rediss"))]:
        try:
            url = urlsplit(values.get(name, ""))
            valid = url.scheme in schemes and url.hostname and url.hostname not in ("localhost", "127.0.0.1", "::1")
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must use a container-reachable host and supported protocol")
    if values.get("KWONREC_ALLOW_UNAUTHENTICATED", "false").lower() != "false":
        raise ValueError("Production requires authentication")
    bind = values.get("KWONREC_BIND_IP", "127.0.0.1")
    try:
        address = ipaddress.IPv4Address(bind)
    except ValueError:
        raise ValueError("KWONREC_BIND_IP must be a private or loopback IPv4 address") from None
    if not (address.is_private or address.is_loopback) or address.is_unspecified or address.is_multicast:
        raise ValueError("KWONREC_BIND_IP must be private or loopback, never a public/wildcard address")
    fingerprint = hashlib.sha256("\n".join(values.get(k, "") for k in
        ("KWONREC_DATABASE_URL", "KWONREC_REDIS_URL", "KWONREC_NAMESPACE")).encode()).hexdigest()
    return bind, fingerprint


if __name__ == "__main__":
    try:
        bind, fingerprint = validate(pathlib.Path(sys.argv[1]).read_text())
        pathlib.Path(sys.argv[2], "binding").write_text(bind)
        pathlib.Path(sys.argv[2], "target").write_text(fingerprint)
    except ValueError as exc:
        sys.exit(str(exc))
