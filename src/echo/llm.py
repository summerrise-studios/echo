"""Language-model backends that return JSON matching a schema.

Two backends, one interface (``LLM.json``):

- ``claude``: the Claude API through the official ``anthropic`` SDK
  (``pip install "<package>[claude]"``). Uses Claude Opus 5.5 with adaptive
  thinking and structured outputs, so the reply always parses. Credentials
  come from the usual places: ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``
  or an ``ant auth login`` profile.
- ``ollama``: a local model served by Ollama (https://ollama.com), called over
  its HTTP API with the standard library only. Free and offline, weaker than
  Claude, good for trying the loop on a laptop.

``pick("auto")`` uses Claude when credentials are configured and falls back to
Ollama when a server is reachable.

This file is shared, unchanged, by Svara, Echo and Frame.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

CLAUDE_MODEL = os.environ.get("SUMMERRISE_CLAUDE_MODEL", "claude-opus-5-5")
OLLAMA_MODEL = os.environ.get("SUMMERRISE_OLLAMA_MODEL", "qwen3:8b")
def _ollama_url(raw: str) -> str:
    """OLLAMA_HOST is often the server's bind address ("0.0.0.0:11434"); make it a URL."""
    raw = (raw or "127.0.0.1:11434").strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    scheme, rest = raw.split("://", 1)
    if rest.startswith("0.0.0.0"):
        rest = "127.0.0.1" + rest[len("0.0.0.0"):]
    if ":" not in rest.split("/", 1)[0]:
        rest = rest.replace("/", ":11434/", 1) if "/" in rest else rest + ":11434"
    return f"{scheme}://{rest}"


OLLAMA_HOST = _ollama_url(os.environ.get("OLLAMA_HOST", ""))


class LLMError(RuntimeError):
    """The model could not be reached or did not return usable JSON."""


@dataclass
class LLM:
    name: str = "base"
    model: str = ""
    calls: int = field(default=0, init=False)

    def json(self, system: str, prompt: str, schema: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} ({self.model})"


@dataclass
class ClaudeLLM(LLM):
    name: str = "claude"
    model: str = CLAUDE_MODEL
    effort: str = "high"
    max_tokens: int = 32000

    def __post_init__(self) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise LLMError('the Claude backend needs the SDK: pip install anthropic') from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        anthropic = self._anthropic
        self.calls += 1
        try:
            # Streaming keeps long structured replies clear of HTTP timeouts.
            # Server-side fallbacks re-run a declined request on another model.
            with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude API: credentials were rejected") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude API: rate limited, try again shortly") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Claude API: could not connect") from e
        if message.stop_reason == "refusal":
            raise LLMError("Claude declined this request")
        if message.stop_reason == "max_tokens":
            raise LLMError("Claude ran out of output room; try a shorter input")
        text = next((b.text for b in message.content if b.type == "text"), "")
        return _parse(text)


@dataclass
class OllamaLLM(LLM):
    name: str = "ollama"
    model: str = OLLAMA_MODEL
    host: str = OLLAMA_HOST
    think: bool = False
    temperature: float = 0.4
    num_ctx: int = 16384
    timeout: float = 900.0
    keep_alive: str | int = "5m"  # Echo sets 0 so the GPU is free for speech models

    def json(self, system: str, prompt: str, schema: dict) -> dict:
        self.calls += 1
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt + "\n\nReply with JSON only."},
            ],
            "format": schema,
            # Streamed, so the timeout is the longest silence between tokens,
            # not the whole reply: a slow laptop model still gets to finish.
            "stream": True,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            _ollama_url(self.host) + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        parts: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("error"):
                        raise LLMError(f"Ollama error: {chunk['error']}")
                    parts.append(chunk.get("message", {}).get("content", ""))
                    if chunk.get("done"):
                        break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"Ollama error {e.code}: {detail}") from e
        except TimeoutError as e:
            raise LLMError(
                f"{self.model} said nothing for {self.timeout:g} s; it may be too big for this "
                "machine. Try a smaller model or --llm claude"
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(f"could not reach Ollama at {self.host} ({e})") from e
        return _parse("".join(parts))


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("the model did not return JSON") from e
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e2:
            raise LLMError("the model returned malformed JSON") from e2
    if not isinstance(data, dict):
        raise LLMError("the model returned JSON that is not an object")
    return data


def claude_configured() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    home = os.path.expanduser("~")
    return os.path.isdir(os.path.join(home, ".config", "anthropic"))


def ollama_available(host: str = OLLAMA_HOST) -> bool:
    try:
        with urllib.request.urlopen(_ollama_url(host) + "/api/tags", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def pick(name: str = "auto", model: str | None = None, **kw) -> LLM:
    """Return a backend by name: ``claude``, ``ollama`` or ``auto``."""
    name = (name or "auto").lower()
    if name == "auto":
        if claude_configured():
            name = "claude"
        elif ollama_available(kw.get("host", OLLAMA_HOST)):
            name = "ollama"
        else:
            raise LLMError(
                "no language model available: set ANTHROPIC_API_KEY for Claude, "
                "or install Ollama and run `ollama pull qwen3:8b`"
            )
    if name == "claude":
        return ClaudeLLM(model=model or CLAUDE_MODEL, **{k: v for k, v in kw.items() if k in ("effort",)})
    if name == "ollama":
        allowed = ("host", "think", "temperature", "num_ctx", "timeout", "keep_alive")
        return OllamaLLM(model=model or OLLAMA_MODEL, **{k: v for k, v in kw.items() if k in allowed})
    raise LLMError(f"unknown backend {name!r}; use claude, ollama or auto")
