import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from rag_core import answer_question


class handler(BaseHTTPRequestHandler):
    def _send_html(self, status_code, html):
        body = html.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        if path in ("", "/"):
            self._send_html(
                200,
                """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medium RAG Assignment API</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.5; color: #111; }
    main { max-width: 760px; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
    a { color: #0645ad; }
  </style>
</head>
<body>
  <main>
    <h1>Medium RAG Assignment API</h1>
    <p>The public API is running.</p>
    <p><strong>Stats:</strong> <a href="/api/stats"><code>GET /api/stats</code></a></p>
    <p><strong>Prompt:</strong> <code>POST /api/prompt</code> with JSON body <code>{"question":"..."}</code></p>
  </main>
</body>
</html>""",
            )
            return

        if path == "/api/stats":
            self._send_json(
                200,
                {
                    "chunk_size": CHUNK_SIZE,
                    "overlap_ratio": OVERLAP_RATIO,
                    "top_k": TOP_K,
                },
            )
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        if path != "/api/prompt":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send_json(400, {"error": "Missing non-empty 'question' field"})
                return

            self._send_json(200, answer_question(question))
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
