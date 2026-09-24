"""Local dev server that emulates the Vercel runtime.

Serves static files from public/ and routes /api/* to the Vercel
serverless handlers in api/ (the exact code deployed to Vercel).

Usage:
    venv/bin/python local/dev_server.py
"""
import importlib
import json
import os
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
API = ROOT / "api"

load_dotenv(ROOT / ".env")

# Ensure repo root is importable so `services.*` resolves.
sys.path.insert(0, str(ROOT))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def _handle_api(self, path: str):
        # Map /api/ask -> api/ask.py, /api/history?id=... -> api/history.py, etc.
        clean_path = path.split("?", 1)[0]
        name = clean_path.split("/")[-1]
        module_name = f"api.{name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"detail": "Not found"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        # Build a raw HTTP request and feed it through a real socket pair so
        # the BaseHTTPRequestHandler subclass runs exactly as on Vercel.
        request_line = f"{self.command} {self.path} HTTP/1.1\r\n".encode()
        header_bytes = b"".join(
            f"{k}: {v}\r\n".encode() for k, v in self.headers.items()
        )
        raw = request_line + header_bytes + b"\r\n" + body

        server_sock, client_sock = socket.socketpair()
        try:
            client_sock.sendall(raw)
            client_sock.shutdown(socket.SHUT_WR)
            handler_cls = module.handler
            handler_cls(server_sock, ("127.0.0.1", 0), self.server)
            response = client_sock.recv(65536)
        finally:
            server_sock.close()
            client_sock.close()

        header_block, _, body_block = response.partition(b"\r\n\r\n")
        status = int(header_block.split(b" ", 2)[1])
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body_block)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api(self.path)
            return
        # Emulate vercel.json rewrites: /question-log -> /question-log.html
        clean = self.path.split("?", 1)[0]
        if clean in ("/question-log", "/bedtime-story"):
            self.path = clean + ".html"
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._handle_api(self.path)
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        if self.path.startswith("/api/"):
            self._handle_api(self.path)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[dev] {fmt % args}\n")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dev server (Vercel emulation) on http://127.0.0.1:{port}")
    print(f"  Landing page: http://127.0.0.1:{port}/")
    server.serve_forever()