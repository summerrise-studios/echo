import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from echo import llm


class FakeOllama(BaseHTTPRequestHandler):
    chunks: list = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for c in self.chunks:
            self.wfile.write((json.dumps(c) + "\n").encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def ollama():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_ollama_reply_is_joined_from_the_stream(ollama):
    FakeOllama.chunks = [
        {"message": {"content": '{"city": '}, "done": False},
        {"message": {"content": '"Paris"}'}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]
    out = llm.OllamaLLM(model="m", host=ollama).json("s", "p", {"type": "object"})
    assert out == {"city": "Paris"}


def test_ollama_stream_error_is_reported(ollama):
    FakeOllama.chunks = [{"error": "model not found"}]
    with pytest.raises(llm.LLMError, match="model not found"):
        llm.OllamaLLM(model="m", host=ollama).json("s", "p", {"type": "object"})
