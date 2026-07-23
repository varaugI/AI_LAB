"""Optional conversational language-model backends.

AI LAB remains usable without either backend: retrieval and extractive answering
still work. A local Ollama server is the recommended path for ChatGPT-like prose
without sending the user's library to a third party.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib import error, request


@dataclass
class BackendResponse:
    text: str
    model: str
    backend: str


class BackendUnavailable(RuntimeError):
    pass


class ChatBackend:
    name = "backend"

    def available(self) -> bool:
        return True

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> BackendResponse:
        raise NotImplementedError


class OllamaBackend(ChatBackend):
    """Talk to a locally running Ollama server through its HTTP API."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and pull model {self.model!r}."
            ) from exc

    def available(self) -> bool:
        try:
            with request.urlopen(self.base_url + "/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> BackendResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
        }
        data = self._post("/api/chat", payload)
        text = str((data.get("message") or {}).get("content", "")).strip()
        if not text:
            raise BackendUnavailable("Ollama returned an empty response.")
        return BackendResponse(text=text, model=self.model, backend=self.name)


class OpenAICompatibleBackend(ChatBackend):
    """Use any OpenAI-compatible local or hosted chat-completions endpoint."""

    name = "openai-compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: int = 120):
        if not model:
            raise ValueError("A model name is required.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = int(timeout)

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> BackendResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendUnavailable(f"Could not reach the configured model server at {self.base_url}.") from exc
        try:
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BackendUnavailable("The model server returned an unexpected response.") from exc
        return BackendResponse(text=text, model=self.model, backend=self.name)


def backend_from_environment() -> ChatBackend | None:
    """Create a backend from environment variables, or return None for extractive mode."""
    backend = os.environ.get("AI_LAB_BACKEND", "").strip().lower()
    if backend in {"", "none", "extractive", "offline"}:
        return None
    if backend == "ollama":
        return OllamaBackend(
            model=os.environ.get("AI_LAB_MODEL", "llama3.2:3b"),
            base_url=os.environ.get("AI_LAB_OLLAMA_URL", "http://127.0.0.1:11434"),
        )
    if backend in {"openai", "openai-compatible", "compatible"}:
        return OpenAICompatibleBackend(
            model=os.environ.get("AI_LAB_MODEL", ""),
            base_url=os.environ.get("AI_LAB_API_BASE", "http://127.0.0.1:1234"),
            api_key=os.environ.get("AI_LAB_API_KEY", ""),
        )
    raise ValueError(f"Unknown AI_LAB_BACKEND: {backend!r}")
