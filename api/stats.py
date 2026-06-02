import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
