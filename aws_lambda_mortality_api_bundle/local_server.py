from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts", "fusion_bundle.joblib"),
)
os.environ.setdefault("API_KEY", "")  # no key required locally

from app.handler import lambda_handler


class LambdaHandler(BaseHTTPRequestHandler):
    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else None

        event = {
            "version": "2.0",
            "rawPath": self.path.split("?")[0],
            "headers": {k.lower(): v for k, v in dict(self.headers).items()},
            "requestContext": {"http": {"method": method}},
            "body": body,
            "isBase64Encoded": False,
        }

        result = lambda_handler(event, None)
        status = result.get("statusCode", 200)
        headers = result.get("headers", {})
        resp_body = result.get("body", "{}").encode("utf-8")

        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[lambda-local] {fmt % args}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Lambda running locally at http://localhost:{port}")
    HTTPServer(("", port), LambdaHandler).serve_forever()
