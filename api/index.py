"""FastAPI backend. On Vercel this file is the serverless function behind /api/*."""
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from subjectmate import rag  # noqa: E402
from subjectmate.llm import PROVIDERS, LLMError, default_provider  # noqa: E402
from subjectmate.retriever import IndexNotBuiltError, load_meta  # noqa: E402

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
# Questions per minute per client IP; protects the server's API keys. 0 disables the limit.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

app = FastAPI(title="SubjectMate API")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    provider: str | None = None
    model: str | None = Field(default=None, max_length=100)
    api_key: str | None = Field(default=None, max_length=300)
    k: int | None = Field(default=None, ge=1, le=10)
    subject: str | None = Field(default=None, max_length=200)
    history: list[Message] = Field(default_factory=list, max_length=12)

    def rag_kwargs(self) -> dict:
        return dict(provider=self.provider, model=self.model, api_key=self.api_key, k=self.k,
                    subject=self.subject, history=[m.model_dump() for m in self.history])


# ---------------------------------------------------------------- rate limiting

_requests: dict[str, deque] = defaultdict(deque)


def check_rate_limit(request: Request) -> None:
    """Sliding one-minute window per client IP (per server instance)."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window = _requests[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, "Too many questions in a short time. Please wait a minute and try again.")
    window.append(now)


# ---------------------------------------------------------------- endpoints

@app.get("/api/health")
def health():
    try:
        meta = load_meta()
    except IndexNotBuiltError as exc:
        return {"status": "no_index", "detail": str(exc)}
    return {"status": "ok", "chunks": meta["num_chunks"], "built_at": meta["built_at"]}


@app.get("/api/documents")
def documents():
    try:
        meta = load_meta()
    except IndexNotBuiltError as exc:
        raise HTTPException(503, str(exc)) from exc
    subjects = sorted({d["subject"] for d in meta["documents"] if d.get("subject")})
    return {"documents": meta["documents"], "subjects": subjects, "embedding_model": meta["embedding_model"]}


@app.get("/api/providers")
def providers():
    return {
        "default": default_provider(),
        "providers": [
            # ready: usable without the user's own key (server key set, or no key needed)
            {"id": p.id, "label": p.label, "default_model": p.default_model,
             "needs_key": p.needs_key, "ready": p.ready(), "key_url": p.key_url}
            for p in PROVIDERS.values()
            if p.id != "ollama" or p.ready()  # list local Ollama only when it is running
        ],
    }


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    check_rate_limit(request)
    try:
        return rag.answer(req.question, **req.rag_kwargs())
    except IndexNotBuiltError as exc:
        raise HTTPException(503, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except RuntimeError as exc:  # e.g. HF_TOKEN missing for query embeddings
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest, request: Request):
    """Newline-delimited JSON events: meta (sources), token (answer text pieces), done, or error."""
    check_rate_limit(request)

    def events():
        try:
            for event in rag.answer_stream(req.question, **req.rag_kwargs()):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except (IndexNotBuiltError, LLMError, RuntimeError) as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Local development only: serve the web UI (on Vercel, public/ is served by the CDN).
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/{name}", include_in_schema=False)
def static_file(name: str):
    path = (PUBLIC_DIR / name).resolve()
    if path.parent != PUBLIC_DIR.resolve() or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
