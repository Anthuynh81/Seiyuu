"""Local attribution provider talking to Ollama.

Two transports, selected by config (``attribution.ollama_transport``):

- ``native`` (default): Ollama's ``/api/chat`` with ``format`` (the JSON schema),
  ``think: false``, and ``options.num_ctx``. This is required for reasoning models such
  as Qwen3 — their thinking tokens otherwise exhaust the context window and the response
  comes back empty (``done_reason: length``). The OpenAI-compatible endpoint exposes no
  way to disable thinking or raise ``num_ctx`` per request, so native is the default.
- ``openai``: the OpenAI SDK pointed at ``/v1`` with ``response_format`` json_schema —
  fine for non-thinking models; kept for parity and easy migration.

The model stays resident between chunks (``keep_alive``, default '5m') for fast cross-chunk
calls; the GPU resource manager forces an explicit ``unload()`` at the attribute->render
handoff so a TTS engine can load without contending for the single GPU (SPEC GPU
discipline). Ollama being unreachable is a clear, actionable error.
"""

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from seiyuu.attribute.providers.base import (
    AttributionError,
    AttributionLLM,
    MalformedOutputError,
)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_TRANSPORTS = ("native", "openai")


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in a ```json fence despite structured-output mode."""
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text


def _parse_json(content: str, model_id: str) -> dict[str, Any]:
    """A flaky local model can emit invalid or duplicated JSON — a retryable failure."""
    try:
        return json.loads(_strip_code_fence(content))
    except json.JSONDecodeError as exc:
        raise MalformedOutputError(f"{model_id!r} returned invalid JSON: {exc}") from exc


def _urllib_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _urllib_get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


PREFLIGHT_TIMEOUT = 3.0  # seconds; /api/tags is a metadata read — never loads a model
_DEFAULT_REGISTRY_PREFIX = "registry.ollama.ai/library/"


class OllamaPreflightError(AttributionError):
    """Ollama isn't ready for an attribution run; the message carries the fix.

    ``code`` is a stable machine string (the API maps it onto the 503 envelope).
    Subclasses AttributionError so every existing CLI/job catch already handles it.
    """

    code = "ollama_not_ready"

    def __init__(self, message: str, *, base_url: str, model: str | None = None) -> None:
        super().__init__(message)
        self.base_url = base_url
        self.model = model


class OllamaUnreachableError(OllamaPreflightError):
    code = "ollama_unreachable"


class OllamaModelMissingError(OllamaPreflightError):
    code = "model_missing"


def ollama_root(base_url: str) -> str:
    """The server root: the native API lives there, not under the /v1 OpenAI shim."""
    return base_url.rstrip("/").removesuffix("/v1")


def canonical_model_name(name: str) -> str:
    """Ollama's own naming: a bare name means ``:latest`` and the default registry/library
    prefix is implicit (``qwen2.5`` == ``qwen2.5:latest`` ==
    ``registry.ollama.ai/library/qwen2.5:latest``). Only the LAST path component can carry
    a tag — a ``host:port/`` registry prefix is not one."""
    name = name.strip().lower().removeprefix(_DEFAULT_REGISTRY_PREFIX)
    last = name.rsplit("/", 1)[-1]
    return name if ":" in last else f"{name}:latest"


def list_ollama_models(
    base_url: str,
    *,
    timeout: float = PREFLIGHT_TIMEOUT,
    get: Callable[..., dict] | None = None,
) -> set[str]:
    """Canonical names of the locally pulled models (GET /api/tags — no generation, so no
    model is loaded onto the GPU). Raises :class:`OllamaUnreachableError` with the fix."""
    get = get or _urllib_get
    url = ollama_root(base_url) + "/api/tags"
    try:
        data = get(url, timeout)
    except (OSError, ValueError) as exc:  # URLError/timeout are OSError; bad JSON ValueError
        raise OllamaUnreachableError(
            f"Ollama isn't reachable at {ollama_root(base_url)} - start it with "
            f"'ollama serve' (or fix OLLAMA_BASE_URL) ({exc})",
            base_url=base_url,
        ) from exc
    names: set[str] = set()
    for entry in data.get("models") or []:
        for key in ("name", "model"):
            if entry.get(key):
                names.add(canonical_model_name(entry[key]))
    return names


def ollama_preflight(
    base_url: str,
    models: list[str] | tuple[str, ...],
    *,
    timeout: float = PREFLIGHT_TIMEOUT,
    get: Callable[..., dict] | None = None,
) -> None:
    """Fail fast before an attribution run: Ollama must answer and every model the run
    will call must already be pulled. One cheap /api/tags read; raises a typed
    :class:`OllamaPreflightError` whose message is the exact fix."""
    installed = list_ollama_models(base_url, timeout=timeout, get=get)
    for model in dict.fromkeys(models):
        if canonical_model_name(model) not in installed:
            raise OllamaModelMissingError(
                f"model {model} isn't pulled in Ollama at {ollama_root(base_url)} - run "
                f"'ollama pull {model}'",
                base_url=base_url,
                model=model,
            )


class OllamaProvider(AttributionLLM):
    provider_id = "local"

    def __init__(
        self,
        *,
        model: str,
        prompts_dir,
        prompt_version: str = "v1",
        emit_thoughts: bool = False,
        base_url: str = "http://localhost:11434/v1",
        transport: str = "native",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        keep_alive: str | int = "5m",
        timeout: float = 600.0,
        unload_poll_timeout: float = 30.0,
        client: Any | None = None,  # OpenAI client (openai transport; injectable for tests)
        post: Callable[..., dict] | None = None,  # native HTTP POST (injectable for tests)
        get: Callable[..., dict] | None = None,  # native HTTP GET (injectable for tests)
    ) -> None:
        super().__init__(
            model=model,
            prompts_dir=prompts_dir,
            prompt_version=prompt_version,
            emit_thoughts=emit_thoughts,
        )
        if transport not in _TRANSPORTS:
            raise ValueError(f"unknown ollama transport {transport!r}; choose from {_TRANSPORTS}")
        self.base_url = base_url
        self.transport = transport
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.unload_poll_timeout = unload_poll_timeout
        self._client = client
        self._post = post or _urllib_post
        self._get = get or _urllib_get
        # The native API lives at the server root, not under the /v1 OpenAI shim.
        self._root = ollama_root(base_url)
        self.native_url = self._root + "/api/chat"

    def _complete_json(
        self, prompt: str, schema: dict[str, Any], attempt: int = 0
    ) -> dict[str, Any]:
        # Retries need variation or they just reproduce the rejected answer; nudge
        # temperature up per attempt while keeping the first pass deterministic.
        temperature = min(0.8, self.temperature + 0.2 * attempt)
        if self.transport == "native":
            return self._complete_native(prompt, schema, temperature)
        return self._complete_openai(prompt, schema, temperature)

    def _complete_native(
        self, prompt: str, schema: dict[str, Any], temperature: float
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            # Disable reasoning: attribution is structured extraction, and a thinking
            # model's <think> block would burn the context window before any JSON.
            "think": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
            "keep_alive": self.keep_alive,
        }
        try:
            data = self._post(self.native_url, payload, self.timeout)
        except urllib.error.URLError as exc:
            raise AttributionError(
                f"cannot reach Ollama at {self.native_url}: is `ollama serve` running and is "
                f"model {self.model_id!r} pulled (`ollama pull {self.model_id}`)? ({exc})"
            ) from exc

        if data.get("done_reason") == "length":
            raise self._truncation_error()
        content = (data.get("message", {}).get("content") or "").strip()
        if not content:
            raise AttributionError(f"Ollama returned empty content for model {self.model_id!r}")
        return _parse_json(content, self.model_id)

    def _complete_openai(
        self, prompt: str, schema: dict[str, Any], temperature: float
    ) -> dict[str, Any]:
        from openai import APIConnectionError

        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key="ollama")
        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "attribution", "schema": schema},
                },
                extra_body={"keep_alive": self.keep_alive},
            )
        except APIConnectionError as exc:
            raise AttributionError(
                f"cannot reach Ollama at {self.base_url}: is `ollama serve` running and is "
                f"model {self.model_id!r} pulled (`ollama pull {self.model_id}`)?"
            ) from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise self._truncation_error()
        content = (choice.message.content or "").strip()
        if not content:
            raise AttributionError(f"Ollama returned empty content for model {self.model_id!r}")
        return _parse_json(content, self.model_id)

    def unload(self) -> None:
        """Free the model from Ollama's VRAM before a TTS engine loads (GPU handoff).

        Ollama's unload is asynchronous and has no /api/unload (0.x): request keep_alive 0 via
        /api/generate, then poll /api/ps until the model is gone. If it never frees, raise
        loudly — proceeding would let two heavy models co-reside and OOM the 8GB card.
        """
        try:
            self._post(
                f"{self._root}/api/generate",
                {"model": self.model_id, "keep_alive": 0, "prompt": ""},
                self.timeout,
            )
        except urllib.error.URLError:
            return  # server unreachable -> nothing is resident to free
        deadline = time.monotonic() + self.unload_poll_timeout
        while True:
            loaded = self._get(f"{self._root}/api/ps", 5.0).get("models", [])
            if not any(m.get("model") == self.model_id for m in loaded):
                return
            if time.monotonic() >= deadline:
                raise AttributionError(
                    f"Ollama did not unload {self.model_id!r} within "
                    f"{self.unload_poll_timeout}s; a TTS engine cannot safely load"
                )
            time.sleep(0.5)

    def _truncation_error(self) -> MalformedOutputError:
        # Retryable, not fatal: a single attempt can run long (e.g. a corrective retry
        # whose extra output overflows the window) while others fit. If every attempt
        # truncates, the chunk is flagged for review carrying this guidance as its reason.
        return MalformedOutputError(
            f"Ollama truncated output for model {self.model_id!r} (hit the context window). "
            f"Raise num_ctx (attribution.num_ctx / OLLAMA_CONTEXT_LENGTH), use a non-thinking "
            f"model, or lower attribution_chunk_tokens."
        )
