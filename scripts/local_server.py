import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from rag_core import answer_question


ROOT = Path(__file__).resolve().parents[1]


class LocalHandler(BaseHTTPRequestHandler):
    def _json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/stats":
            self._json(
                200,
                {
                    "chunk_size": CHUNK_SIZE,
                    "overlap_ratio": OVERLAP_RATIO,
                    "top_k": TOP_K,
                },
            )
            return

        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/prompt":
            self._json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json(400, {"error": "Missing non-empty 'question' field"})
                return
            self._json(200, answer_question(question))
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), LocalHandler)
    print("Local server running at http://127.0.0.1:8000", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
