"""Serve the KYC demo from this computer at a public HTTPS address (free).

    python -m scripts.public_demo [--tunnel ngrok|cloudflare] [--domain D] [--port 8080]
                                  [--cors https://your-site.vercel.app] [--adjustable-checks]

Starts the API the way a public server must run - access codes on, debug
off, one worker - on 127.0.0.1 only, then opens a tunnel to it and prints the
public HTTPS address. Phone cameras work there because it is HTTPS. Ctrl+C
stops both.

Tunnels:
- ngrok (default): runs over port 443. Needs a free account; add its authtoken once, in your own terminal:
      .data/tools/ngrok/ngrok.exe config add-authtoken <token>
  The free plan gives one fixed address (dashboard > Domains); pass it as
  --domain to keep the same URL across restarts. Browsers see ngrok's warning
  page once; the site's API client sends `ngrok-skip-browser-warning`.
- cloudflare: a quick tunnel (https://<random>.trycloudflare.com), no account,
  but it needs outbound TCP/UDP 7844 to Cloudflare, and the address changes
  on every start.
- The first run makes a demo access code in .data/deploy/access_codes.json
  (git-ignored); give it to testers. Without a code every /v1 call is 401.
- Card photos and selfies are processed on this machine only; Cloudflare
  carries the encrypted traffic.
- ngrok / cloudflared are looked for in .data/tools/<name>/ and then on PATH
  (https://ngrok.com/download, https://github.com/cloudflare/cloudflared/releases).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / ".data" / "deploy"
CODES_FILE = DEPLOY_DIR / "access_codes.json"
URL_FILE = DEPLOY_DIR / "public_url.txt"
# A quick tunnel is <words>-<words>.trycloudflare.com; "api.trycloudflare.com"
# (its control endpoint) also appears in error lines and is not one.
TUNNEL_URL = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")
# Not passed to either tunnel (both are Go programs). SSL_CERT_FILE: Go reads
# its trusted roots from it, and on this machine it points at openssl.cnf (not
# a certificate list), so every certificate fails ("signed by unknown
# authority"); without it Go uses the Windows store.
CERT_VARS = ("SSL_CERT_FILE", "SSL_CERT_DIR")
# Proxies are not passed either: cloudflared does not use them, and ngrok's
# free plan refuses to start through one ("a Pay-as-you-go feature"). Both
# need a direct route out; from Iran that means a VPN in TUN/VPN mode, since a
# system-proxy-only VPN leaves them on the Iranian IP, which ngrok refuses
# (ERR_NGROK_9040) and which cannot reach Cloudflare's port 7844.
PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


def access_codes() -> dict[str, str]:
    if CODES_FILE.exists():
        return json.loads(CODES_FILE.read_text(encoding="utf-8"))
    codes = {"demo": secrets.token_urlsafe(9)}
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    CODES_FILE.write_text(json.dumps(codes, indent=2) + "\n", encoding="utf-8")
    print(f"made a demo access code: {CODES_FILE.relative_to(REPO_ROOT)}")
    return codes


def tool(name: str) -> str:
    local = REPO_ROOT / ".data" / "tools" / name / (f"{name}.exe" if os.name == "nt" else name)
    found = str(local) if local.exists() else shutil.which(name)
    if not found:
        sys.exit(f"{name} not found: put it in .data/tools/{name}/")
    return found


def tunnel_command(kind: str, port: int, domain: str | None) -> tuple[list[str], dict[str, str]]:
    env = {k: v for k, v in os.environ.items() if k not in CERT_VARS + PROXY_VARS}
    if kind == "cloudflare":
        return [tool("cloudflared"), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], env
    cmd = [tool("ngrok"), "http", f"127.0.0.1:{port}", "--log", "stdout", "--log-format", "json"]
    if domain:
        cmd += ["--url", domain if "://" in domain else f"https://{domain}"]
    return cmd, env


def tunnel_event(kind: str, line: str) -> tuple[str | None, str | None]:
    """(public url, error text) found in one line of the tunnel's log."""
    if kind == "cloudflare":
        m = TUNNEL_URL.search(line)
        error = line.rstrip() if (" ERR " in line or line.startswith("failed")) else None
        return (m.group(0) if m else None), error
    try:
        event = json.loads(line)
    except ValueError:
        return None, line.rstrip() or None
    url = event.get("url") if event.get("msg") == "started tunnel" else None
    error = event.get("err") if event.get("lvl") in ("eror", "crit") else None
    return url, (str(error) if error and error != "<nil>" else None)


def wait_for_health(port: int, server: subprocess.Popen, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            sys.exit(f"the API exited with code {server.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return
        except OSError:
            time.sleep(1)
    sys.exit("the API did not answer /health in time")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tunnel", choices=("ngrok", "cloudflare"), default="ngrok")
    p.add_argument("--domain", help="ngrok: your free fixed domain (xxx.ngrok-free.app)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--cors", nargs="*", default=[], help="Site origins allowed to call the API")
    p.add_argument(
        "--adjustable-checks",
        action="store_true",
        help="Let testers switch selfie checks off (KYC_ADJUSTABLE_CHECKS) - test demos only",
    )
    args = p.parse_args()

    env = os.environ | {
        "KYC_REQUIRE_ACCESS": "true",
        "KYC_DEBUG": "false",
        "KYC_ADJUSTABLE_CHECKS": "true" if args.adjustable_checks else "false",
        "KYC_ACCESS_CODES": json.dumps(access_codes()),
        "KYC_CORS_ORIGINS": json.dumps(args.cors),
    }
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "kyc.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workers",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    tunnel = None
    try:
        wait_for_health(args.port, server)
        print(f"API up on 127.0.0.1:{args.port} (access codes on, debug off); opening the tunnel...")
        cmd, env = tunnel_command(args.tunnel, args.port, args.domain)
        tunnel = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        url = None
        for line in tunnel.stdout:
            found, error = tunnel_event(args.tunnel, line)
            if url is None and found:
                url = found
                URL_FILE.write_text(url + "\n", encoding="utf-8")
                print(
                    f"\n  public address: {url}\n  (also in {URL_FILE.relative_to(REPO_ROOT)}) - Ctrl+C to stop\n",
                    flush=True,
                )
            elif error:
                print(f"{args.tunnel}: {error}", flush=True)
        sys.exit(f"the tunnel exited with code {tunnel.wait()}")
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (tunnel, server):
            if proc and proc.poll() is None:
                proc.terminate()
        URL_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
