"""Pluggable LLM client for the multi-agent system.

Default backend is a LOCAL model served by Ollama (free, offline, no API key). An
OpenAI-compatible HTTP backend can be enabled by setting environment variables, so the same
agent code can later be pointed at a stronger hosted model without changes:

    ZDV_LLM_BACKEND=ollama|openai      (default: ollama)
    ZDV_LLM_MODEL=qwen2.5:7b           (default per backend)
    ZDV_LLM_BASE_URL=http://...        (openai backend only)
    ZDV_LLM_API_KEY=...                (openai backend only)

Every call is recorded with its latency so the evaluation can report time-to-decision, and
generation is greedy (temperature 0, fixed seed) to keep runs as reproducible as LLMs allow.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class LLMCall:
    """One request/response pair, kept for the audit trail and timing metrics."""

    prompt: str
    response: str
    seconds: float
    model: str


@dataclass
class LLMStats:
    calls: list[LLMCall] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def total_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Minimal chat client with a single `chat()` entry point."""

    def __init__(self, backend: str | None = None, model: str | None = None,
                 temperature: float = 0.0, seed: int = 17, timeout: int = 180) -> None:
        self.backend = (backend or os.getenv("ZDV_LLM_BACKEND", "ollama")).lower()
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        if self.backend == "ollama":
            self.model = model or os.getenv("ZDV_LLM_MODEL", DEFAULT_OLLAMA_MODEL)
            self.base_url = os.getenv("ZDV_LLM_BASE_URL", DEFAULT_OLLAMA_URL)
            self.api_key = None
        elif self.backend == "anthropic":
            # Anthropic's Messages API is not OpenAI-compatible: different endpoint, different
            # auth header, a top-level `system` field, and a required max_tokens.
            self.model = model or os.getenv("ZDV_LLM_MODEL", "claude-sonnet-5")
            self.base_url = os.getenv("ZDV_LLM_BASE_URL", "https://api.anthropic.com/v1")
            self.api_key = os.getenv("ZDV_LLM_API_KEY")
            self.max_tokens = int(os.getenv("ZDV_LLM_MAX_TOKENS", "200"))
            if not self.api_key:
                raise LLMError("anthropic backend selected but ZDV_LLM_API_KEY is not set")
        elif self.backend == "openai":
            self.model = model or os.getenv("ZDV_LLM_MODEL", "gpt-4o-mini")
            self.base_url = os.getenv("ZDV_LLM_BASE_URL", "https://api.openai.com/v1")
            self.api_key = os.getenv("ZDV_LLM_API_KEY")
            if not self.api_key:
                raise LLMError("openai backend selected but ZDV_LLM_API_KEY is not set")
        else:
            raise LLMError(f"unknown backend: {self.backend}")
        self.stats = LLMStats()

    # -- public API ----------------------------------------------------------------------

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send a system+user prompt, return the assistant text."""
        t0 = time.perf_counter()
        if self.backend == "ollama":
            text = self._chat_ollama(system, user, json_mode)
        elif self.backend == "anthropic":
            text = self._chat_anthropic(system, user, json_mode)
        else:
            text = self._chat_openai(system, user, json_mode)
        dt = time.perf_counter() - t0
        self.stats.calls.append(LLMCall(prompt=user, response=text, seconds=dt, model=self.model))
        return text

    def chat_json(self, system: str, user: str, retries: int = 2) -> dict:
        """Chat and parse a JSON object, retrying once with a stricter nudge on failure."""
        for _ in range(retries + 1):
            raw = self.chat(system, user, json_mode=True)
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            user = (f"{user}\n\nTwoja poprzednia odpowiedź nie była poprawnym JSON-em. "
                    f"Odpowiedz WYŁĄCZNIE jednym obiektem JSON, bez komentarzy.")
        raise LLMError(f"model did not return valid JSON after {retries + 1} attempts")

    def available(self) -> bool:
        """True if the backend answers (used to fail fast with a helpful message)."""
        try:
            if self.backend == "ollama":
                with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as r:
                    return r.status == 200
            return bool(self.api_key)
        except Exception:
            return False

    # -- backends ------------------------------------------------------------------------

    def _chat_ollama(self, system: str, user: str, json_mode: bool) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if json_mode:
            payload["format"] = "json"
        data = self._post(f"{self.base_url}/api/chat", payload)
        return data.get("message", {}).get("content", "")

    def _chat_anthropic(self, system: str, user: str, json_mode: bool) -> str:
        """Anthropic Messages API. `max_tokens` is required and kept small: every prompt in this
        project asks for one short JSON object, and a low cap bounds cost."""
        prompt = user
        if json_mode:
            prompt += "\n\nRespond with the JSON object only, no prose and no code fence."
        # `temperature` is rejected by current Anthropic models, so it is not sent. Unlike the
        # local backends, this one is therefore NOT pinned to greedy decoding, which is worth
        # stating wherever its results are reported.
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = self._post(f"{self.base_url}/messages", payload, headers)
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        usage = data.get("usage", {})
        self.tokens_in = getattr(self, "tokens_in", 0) + int(usage.get("input_tokens", 0))
        self.tokens_out = getattr(self, "tokens_out", 0) + int(usage.get("output_tokens", 0))
        return "".join(parts)

    def _chat_openai(self, system: str, user: str, json_mode: bool) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post(f"{self.base_url}/chat/completions", payload, headers)
        return data["choices"][0]["message"]["content"]

    def _post(self, url: str, payload: dict, headers: dict | None = None,
              retries: int = 3) -> dict:
        """POST with a short retry. A local server occasionally returns a transient 500;
        without a retry a single such response aborts an evaluation run of several hours."""
        body = json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=body, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.URLError as e:
                last = e
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise LLMError(
            f"LLM backend '{self.backend}' unreachable at {url}: {last}. "
            f"For the local backend start it with `ollama serve` and "
            f"`ollama pull {self.model}`.") from last


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from a model response, tolerating stray prose or code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start < 0:
        return None
    for i in range(start, len(text)):  # first balanced {...} block
        depth += 1 if text[i] == "{" else (-1 if text[i] == "}" else 0)
        if depth == 0:
            try:
                obj = json.loads(text[start:i + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None
