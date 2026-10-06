"""Answer generation with a choice of LLM provider.

Each provider is called through its official SDK. The API key comes from the request
(a user's own key) or, if absent, from the server's environment variables. Two options need
no key: a local Ollama model, and "extractive" mode (no LLM; see extractive.py).
"""
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    env_keys: tuple[str, ...]
    default_model: str
    key_url: str
    needs_key: bool = True

    def server_key(self) -> str | None:
        for name in self.env_keys:
            if os.getenv(name):
                return os.getenv(name)
        return None

    def ready(self, api_key: str | None = None) -> bool:
        """Whether this provider can answer now (with the given user key, if any)."""
        if self.id == "ollama":
            return ollama_running()
        if not self.needs_key:
            return True
        return bool((api_key or "").strip() or self.server_key())


PROVIDERS: dict[str, Provider] = {
    p.id: p
    for p in [
        Provider("anthropic", "Claude (Anthropic)", ("ANTHROPIC_API_KEY",),
                 os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"), "https://console.anthropic.com/settings/keys"),
        Provider("openai", "OpenAI", ("OPENAI_API_KEY",),
                 os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "https://platform.openai.com/api-keys"),
        Provider("groq", "Groq (Llama)", ("GROQ_API_KEY",),
                 os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "https://console.groq.com/keys"),
        Provider("gemini", "Google Gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                 os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "https://aistudio.google.com/apikey"),
        Provider("ollama", "Ollama (local, free)", (),
                 os.getenv("OLLAMA_MODEL", "llama3.2:3b"), "https://ollama.com/download", needs_key=False),
        Provider("extractive", "No LLM (extract from notes)", (),
                 "sentence ranking", "", needs_key=False),
    ]
}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def ollama_running() -> bool:
    """True if a local Ollama server answers (checked once per process; never on Vercel)."""
    if os.getenv("VERCEL"):
        return False
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1):
            return True
    except OSError:
        return False


def default_provider() -> str:
    """LLM_PROVIDER if usable, else the first provider with a server key, else Ollama, else extractive."""
    preferred = PROVIDERS.get(os.getenv("LLM_PROVIDER", ""))
    if preferred and preferred.ready():
        return preferred.id
    for provider in PROVIDERS.values():
        if provider.needs_key and provider.server_key():
            return provider.id
    return "ollama" if ollama_running() else "extractive"


def _resolve(provider: str | None, model: str | None, api_key: str | None) -> tuple[Provider, str, str]:
    provider_id = provider or default_provider()
    if provider_id not in PROVIDERS:
        raise LLMError(f"Unknown provider {provider_id!r}.", status=400)
    p = PROVIDERS[provider_id]
    if provider_id == "extractive":
        raise LLMError("Extractive mode does not use an LLM.", status=400)
    key = (api_key or "").strip() or p.server_key() or ""
    if p.needs_key and not key:
        raise LLMError(
            f"No API key for {p.label}. Add your own key in Settings or choose another model.",
            status=400,
        )
    return p, (model or "").strip() or p.default_model, key


def stream(system: str, user: str, provider: str | None = None, model: str | None = None,
           api_key: str | None = None) -> Iterator[str]:
    """Yield the answer text in pieces as the provider generates it."""
    p, model, key = _resolve(provider, model, api_key)
    caller = {"anthropic": _anthropic, "openai": _openai, "groq": _groq, "gemini": _gemini,
              "ollama": _ollama}[p.id]
    try:
        yield from caller(system, user, model, key)
    except LLMError:
        raise
    except Exception as exc:  # SDK-specific errors from the non-Anthropic providers
        raise LLMError(f"{p.label} request failed: {exc}") from exc


def generate(system: str, user: str, provider: str | None = None,
             model: str | None = None, api_key: str | None = None) -> tuple[str, str, str]:
    """Return (answer_text, provider_id, model)."""
    p, model_used, _ = _resolve(provider, model, api_key)
    text = "".join(stream(system, user, p.id, model_used, api_key))
    return text.strip(), p.id, model_used


def resolved_model(provider: str, model: str | None) -> str:
    return (model or "").strip() or PROVIDERS[provider].default_model


# Models that support the effort setting and server-side refusal fallbacks.
_CLAUDE_CURRENT = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


def _anthropic(system: str, user: str, model: str, key: str) -> Iterator[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=55, max_retries=1)
    params = dict(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    try:
        if model in _CLAUDE_CURRENT:
            # Grounded Q&A over short excerpts does not need deep reasoning; "low" effort keeps
            # latency well inside the serverless time limit. "fallbacks" re-runs a request
            # declined by a safety classifier on Anthropic's recommended fallback model.
            manager = client.beta.messages.stream(
                **params,
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default", "output_config": {"effort": "low"}},
            )
        else:
            manager = client.messages.stream(**params)
        with manager as response:
            yield from response.text_stream
            final = response.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise LLMError("The Anthropic API key was rejected.", status=401) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic rate limit reached. Please try again shortly.", status=429) from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API.") from exc

    if final.stop_reason == "refusal":
        raise LLMError("Claude declined to answer this question.", status=422)
    if final.stop_reason == "max_tokens":
        yield "\n\n(Answer truncated.)"


def _openai(system: str, user: str, model: str, key: str) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=55, max_retries=1)
    chunks = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=True,
    )
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _groq(system: str, user: str, model: str, key: str) -> Iterator[str]:
    from groq import Groq

    client = Groq(api_key=key, timeout=55, max_retries=1)
    chunks = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=True,
    )
    for chunk in chunks:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _gemini(system: str, user: str, model: str, key: str) -> Iterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0),
    ):
        if chunk.text:
            yield chunk.text


def _ollama(system: str, user: str, model: str, key: str) -> Iterator[str]:
    import ollama

    client = ollama.Client(host=OLLAMA_HOST, timeout=180)
    try:
        chunks = client.chat(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0},
            stream=True,
        )
        thinking = False  # reasoning models (qwen3, deepseek-r1) wrap their reasoning in <think> tags
        for chunk in chunks:
            text = chunk["message"]["content"]
            if "<think>" in text:
                thinking, text = True, text.split("<think>")[0]
            if thinking:
                if "</think>" not in text:
                    continue
                thinking, text = False, text.split("</think>", 1)[1]
            if text:
                yield text
    except ollama.ResponseError as exc:
        hint = f" Run: ollama pull {model}" if exc.status_code == 404 else ""
        raise LLMError(f"Ollama error: {exc.error}.{hint}") from exc
