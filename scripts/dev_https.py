"""Serve the API over HTTPS on the local network, to test the dev page from a phone.

    python -m scripts.dev_https            # make the certificate, print the command
    python -m scripts.dev_https --run      # ...and start the server

Phone browsers allow camera access only on HTTPS (plain HTTP works for
localhost alone), so testing with a real phone camera needs this. It writes a
self-signed certificate for this machine's LAN IP to .data/dev_tls/ (not in
git); the phone shows a certificate warning once, which you accept.

FOR TESTING ONLY: the server runs with KYC_DEBUG=true and listens on the
local network. Stop it (Ctrl+C) when done. Windows may ask whether Python may
accept connections - allow "Private networks" only.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

TLS_DIR = Path(".data/dev_tls")


def lan_ip() -> str:
    """The address other devices on this network reach us at (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def make_cert(ip: str) -> tuple[Path, Path]:
    openssl = shutil.which("openssl")
    if not openssl:
        raise SystemExit("openssl not found (it ships with Git for Windows: run this from Git Bash)")
    ipaddress.ip_address(ip)  # refuse anything that is not an IP
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    key, cert = TLS_DIR / "key.pem", TLS_DIR / "cert.pem"
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30",
         "-keyout", str(key), "-out", str(cert), "-subj", "/CN=kyc-dev",
         "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
        check=True, capture_output=True,
    )
    return key, cert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--ip", default=None, help="LAN IP to put in the certificate (default: detected)")
    parser.add_argument("--run", action="store_true", help="start the server after making the certificate")
    args = parser.parse_args()

    ip = args.ip or lan_ip()
    key, cert = make_cert(ip)
    cmd = [sys.executable, "-m", "uvicorn", "kyc.api.app:app", "--host", "0.0.0.0", "--port", str(args.port),
           "--ssl-keyfile", str(key), "--ssl-certfile", str(cert)]
    print(f"certificate: {cert} (valid 30 days, for {ip})")
    print(f"open on the phone (same Wi-Fi):  https://{ip}:{args.port}/dev/face-capture")
    print("the phone will warn about the certificate - accept it once (Advanced -> proceed)")
    if args.run:
        env = {**os.environ, "KYC_DEBUG": "true"}
        print("starting - Ctrl+C to stop")
        subprocess.run(cmd, env=env, check=False)
    else:
        print("start the server with KYC_DEBUG=true and:\n  " + " ".join(cmd))


if __name__ == "__main__":
    main()
