import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_core import answer_question


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
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

    def do_GET(self):
        self._send_json(405, {"error": "Use POST with JSON body: {'question': '...'}"})
