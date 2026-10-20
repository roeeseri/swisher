"""Generate a short-lived local CA and HTTPS endpoint for a paired phone.

Nothing is installed into an OS trust store automatically. The owner must install
and trust the certificate on their own phone before browser camera capture works.
"""

from __future__ import annotations

import plistlib
import socket
import ssl
import subprocess
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4


def lan_ip():
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("192.0.2.1", 9))
        address = connection.getsockname()[0]
        if address.startswith("127."):
            raise ValueError("לא נמצא חיבור רשת מקומי")
        return address
    finally:
        connection.close()


def certificate(root: Path, address: str):
    root.mkdir(parents=True, exist_ok=True)
    config = root / "server.cnf"
    config.write_text(f"""[req]
distinguished_name = dn
prompt = no
[dn]
CN = Swisher local computer
[ext]
subjectAltName = IP:{address},IP:127.0.0.1,DNS:localhost
extendedKeyUsage = serverAuth
keyUsage = digitalSignature,keyEncipherment
basicConstraints = CA:FALSE
""")

    def run(*args):
        subprocess.run(["openssl", *args], check=True, capture_output=True)

    if not (root / "ca.key").exists():
        run(
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "30",
            "-keyout",
            str(root / "ca.key"),
            "-out",
            str(root / "ca.crt"),
            "-subj",
            "/CN=Swisher Local Pilot CA",
            "-sha256",
        )
    run(
        "req",
        "-new",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(root / "server.key"),
        "-out",
        str(root / "server.csr"),
        "-config",
        str(config),
    )
    run(
        "x509",
        "-req",
        "-in",
        str(root / "server.csr"),
        "-CA",
        str(root / "ca.crt"),
        "-CAkey",
        str(root / "ca.key"),
        "-CAcreateserial",
        "-out",
        str(root / "server.crt"),
        "-days",
        "14",
        "-sha256",
        "-extfile",
        str(config),
        "-extensions",
        "ext",
    )
    for path in root.glob("*.key"):
        path.chmod(0o600)
    der = ssl.PEM_cert_to_DER_cert((root / "ca.crt").read_text())
    profile = {
        "PayloadContent": [
            {
                "PayloadCertificateFileName": "Swisher-Local-CA.cer",
                "PayloadContent": der,
                "PayloadDescription": "תעודה לחיבור מקומי למחשב Swisher בלבד",
                "PayloadDisplayName": "Swisher Local Pilot CA",
                "PayloadIdentifier": "local.swisher.ca",
                "PayloadType": "com.apple.security.root",
                "PayloadUUID": str(uuid4()),
                "PayloadVersion": 1,
            }
        ],
        "PayloadDisplayName": "Swisher — חיבור למחשב האישי",
        "PayloadDescription": "תעודת אמון מקומית לפיילוט. אין ניהול מכשיר ואין שרת חיצוני. להסרה אחרי הפיילוט.",
        "PayloadIdentifier": "local.swisher.profile",
        "PayloadOrganization": "Swisher local project",
        "PayloadRemovalDisallowed": False,
        "PayloadType": "Configuration",
        "PayloadUUID": str(uuid4()),
        "PayloadVersion": 1,
        "RemovalDate": datetime.now(UTC).replace(tzinfo=None) + timedelta(days=30),
    }
    (root / "Swisher.mobileconfig").write_bytes(plistlib.dumps(profile))
    return root / "server.crt", root / "server.key"


class CertificateHandler(BaseHTTPRequestHandler):
    """Bootstrap serves only public certificate files, never videos or keys."""

    def do_GET(self):
        filenames = {"/Swisher.mobileconfig": "Swisher.mobileconfig", "/Swisher-CA.crt": "ca.crt"}
        if self.path == "/":
            content = """<!doctype html><html dir="rtl" lang="he"><meta name="viewport" content="width=device-width,initial-scale=1"><meta charset="utf-8"><title>חיבור Swisher</title><body style="font:18px system-ui;padding:24px;line-height:1.7"><h1>חיבור הטלפון למחשב Swisher</h1><p>זהו שלב הגדרה חד־פעמי ברשת הביתית. אין וידאו בדף זה.</p><p>התקנת תעודת שורש מוסיפה אמון במחשב הזה. בצעו זאת רק בטלפון שלכם, ברשת מהימנה, והסירו את הפרופיל בתום הפיילוט.</p><h2>iPhone</h2><ol><li><a href="/Swisher.mobileconfig">הורדת פרופיל Swisher</a></li><li>בהגדרות: פרופיל שהורד ← התקנה.</li><li>הגדרות ← כללי ← אודות ← הגדרות אמון בתעודות ← הפעלת אמון עבור Swisher Local Pilot CA.</li><li>פתחו את כתובת הצילום המאובטחת שמופיעה במסך המחשב והזינו קוד צימוד.</li></ol><h2>Android</h2><p><a href="/Swisher-CA.crt">הורדת תעודת Swisher</a>. התקינו כתעודת CA בהגדרות האבטחה, ואז פתחו את כתובת הצילום.</p><p>אל תעקפו אזהרת אבטחה. אם יש אזהרה גם אחרי ההגדרה, חזרו למחשב ובדקו את החיבור.</p></body></html>""".encode()
            mime = "text/html; charset=utf-8"
        elif self.path in filenames:
            content = (self.server.certificate_root / filenames[self.path]).read_bytes()
            mime = (
                "application/x-apple-aspen-config"
                if self.path.endswith("mobileconfig")
                else "application/x-x509-ca-cert"
            )
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


def bootstrap_server(root: Path, port: int):
    server = ThreadingHTTPServer(("0.0.0.0", port), CertificateHandler)
    server.certificate_root = root
    return server
