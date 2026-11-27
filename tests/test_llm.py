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


class FakeGemini(BaseHTTPRequestHandler):
    replies: list = []
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGemini.seen.append((self.headers.get("x-goog-api-key"), body))
        code, reply = FakeGemini.replies.pop(0)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def gemini(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGemini)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm, "GEMINI_URL", f"http://127.0.0.1:{httpd.server_address[1]}/{{model}}")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    FakeGemini.seen = []
    yield
    httpd.shutdown()


def test_gemini_sends_the_schema_and_skips_thoughts(gemini):
    FakeGemini.replies = [(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": '{"city": "Paris"}'}]}}]})]
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    out = llm.GeminiLLM(model="m").json("sys", "p", schema)
    assert out == {"city": "Paris"}
    key, body = FakeGemini.seen[0]
    assert key == "test-key"
    assert body["generationConfig"]["responseJsonSchema"] == schema
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_retries_when_busy_and_stops_on_daily_quota(gemini, monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    ok = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
    FakeGemini.replies = [(503, {"error": {"message": "busy"}}), (200, ok)]
    assert llm.GeminiLLM(model="m").json("s", "p", {"type": "object"}) == {}
    FakeGemini.replies = [(429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})]
    with pytest.raises(llm.LLMError, match="quota"):
        llm.GeminiLLM(model="m").json("s", "p", {"type": "object"})
