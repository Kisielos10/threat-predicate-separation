"""Minimal target web service for the mock environment (standard library only).

Serves a handful of endpoints and appends every request to /logs/requests.jsonl. It is a
*target*, not a detector: it records what was requested and does not judge it. The X-Case-Id
header is recorded for evaluation bookkeeping only and is never shown to the agents (they see
method, path, query and body).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

LOG_PATH = os.environ.get("LOG_PATH", "/logs/requests.jsonl")
PORT = int(os.environ.get("PORT", "8081"))

PAGES = {
    "/": "<h1>Sklep</h1><a href='/products'>Produkty</a> <a href='/login'>Logowanie</a>",
    "/products": "<h1>Produkty</h1><ul><li>Kawa</li><li>Herbata</li></ul>",
    "/login": "<form method='POST' action='/login'><input name='user'><input name='pass'></form>",
    "/search": "<h1>Wyniki wyszukiwania</h1>",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "MockShop/1.0"

    def _log(self, method: str, body: str) -> None:
        parsed = urlparse(self.path)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "client": self.client_address[0],
            "method": method,
            "path": parsed.path,
            "query": parsed.query,
            "body": body,
            "case_id": self.headers.get("X-Case-Id", ""),
            "user_agent": self.headers.get("User-Agent", ""),
        }
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _respond(self, path: str) -> None:
        body = PAGES.get(path)
        status = 200 if body else 404
        payload = (body or "<h1>404</h1>").encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._log("GET", "")
        self._respond(urlparse(self.path).path)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
        self._log("POST", body)
        self._respond(urlparse(self.path).path)

    def log_message(self, *args) -> None:  # silence stderr access logs
        return


if __name__ == "__main__":
    print(f"mock target listening on :{PORT}, logging to {LOG_PATH}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
