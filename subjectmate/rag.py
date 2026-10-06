"""The RAG pipeline: retrieve course excerpts, build a grounded prompt, generate a cited answer."""
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .extractive import extractive_answer
from .llm import PROVIDERS, LLMError, default_provider, generate, resolved_model, stream
from .retriever import get_retriever, min_score

SYSTEM_PROMPT = (
    "You are SubjectMate, a study assistant for university students. You answer their questions "
    "using only the excerpts from the course material (lecture slides, notes, practice "
    "sets and papers) that come with each question.\n\n"
    "Rules:\n"
    "- Base the answer only on the excerpts. Do not add facts from outside knowledge, even if you "
    "know them; students need answers they can verify in their course material.\n"
    "- After each claim, cite its source in square brackets with the file name and location "
    "exactly as written in the excerpt label, for example [Lecture_03.pdf, p. 12] or [Regression.pptx, slide 4].\n"
    "- If the excerpts do not contain the answer, reply with exactly this sentence: "
    f'"{config.NOT_FOUND}" You may then add one sentence naming related topics the excerpts do cover.\n'
    "- If the excerpts answer only part of the question, answer that part and say what is missing.\n"
    "- Explain clearly for a student. Use short paragraphs, and bullet points or numbered steps "
    "where they help. Use Markdown for formatting, and write formulas in LaTeX between $...$ "
    "(inline) or $$...$$ (on their own line)."
)

BASELINE_SYSTEM_PROMPT = (
    "You are a study assistant for a university course. Answer the student's question clearly "
    "and concisely."
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT),
     ("human", "{history}Course excerpts:\n\n{context}\n\nQuestion: {question}")]
)

REWRITE_PROMPT = (
    "Rewrite the student's follow-up question as one standalone question that can be understood "
    "without the conversation, keeping the topic it refers to. Reply with the question only."
)

HISTORY_TURNS = 3  # previous question/answer pairs used for follow-ups


def _location(d: Document) -> str:
    return f"{d.metadata['source']}, {d.metadata.get('unit', 'p.')} {d.metadata['page']}"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        # Label each excerpt with its exact citation so models copy it rather than "[1]".
        f"Excerpt from [{_location(d)}]:\n{d.page_content}"
        for d in docs
    )


def _sources(docs: list[Document]) -> list[dict]:
    """One entry per (file, page), best score first."""
    best: dict[tuple[str, int], dict] = {}
    for d in docs:
        key = (d.metadata["source"], d.metadata["page"])
        if key not in best or d.metadata["score"] > best[key]["score"]:
            best[key] = {
                "source": key[0],
                "subject": d.metadata.get("subject", ""),
                "unit": d.metadata.get("unit", "p."),
                "page": key[1],
                "score": round(d.metadata["score"], 3),
                "snippet": d.page_content[:400],
            }
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)


_CITATION = re.compile(
    r"\s?\[([^\[\]]+?\.(?:pdf|pptx|docx|ipynb|md|txt)),\s*(p\.|pp\.|slides?|parts?)\s*([\d\s,–-]+)\]",
    re.IGNORECASE)


def remove_unsupported_citations(text: str, docs: list[Document]) -> str:
    """Drop citations that do not point to a retrieved passage (LLMs sometimes invent sources)."""
    retrieved = {(d.metadata["source"].lower(), d.metadata["page"]) for d in docs}

    def keep(match: re.Match) -> str:
        source = match.group(1).strip().lower()
        pages = [int(p) for p in re.findall(r"\d+", match.group(3))]
        ok = any((source, p) in retrieved for p in pages) or (
            any(source == s for s, _ in retrieved) and len(pages) > 1)  # page ranges in a retrieved file
        return match.group(0) if ok else ""

    return _CITATION.sub(keep, text)


# ---------------------------------------------------------------- follow-up questions

_FOLLOW_UP = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|above|previous|same|former|latter|"
    r"example|another|else|more|again)\b|^\s*(and|but|so|why|how about|what about|then|also)\b",
    re.IGNORECASE,
)


def _last_turns(history: list[dict] | None) -> list[dict]:
    turns = [m for m in (history or []) if m.get("role") in {"user", "assistant"} and m.get("content")]
    return turns[-2 * HISTORY_TURNS:]


def is_follow_up(question: str, history: list[dict] | None) -> bool:
    has_previous = any(m["role"] == "user" for m in _last_turns(history))
    return has_previous and (len(question.split()) <= 4 or bool(_FOLLOW_UP.search(question)))


def standalone_question(question: str, history: list[dict] | None, provider: str,
                        model: str | None, api_key: str | None) -> str:
    """Turn a follow-up ("what is its time complexity?") into a question retrieval can use."""
    if not is_follow_up(question, history):
        return question
    turns = _last_turns(history)
    previous = next(m["content"] for m in reversed(turns) if m["role"] == "user")
    # A hosted LLM rewrites the question properly; offline (local model / no LLM) we keep it fast
    # and simply add the previous question for context.
    if provider not in {"ollama", "extractive"}:
        convo = "\n".join(f"{m['role'].title()}: {m['content'][:500]}" for m in turns)
        try:
            text, _, _ = generate(REWRITE_PROMPT, f"Conversation:\n{convo}\n\nFollow-up: {question}",
                                  provider, model, api_key)
            if text and len(text) < 300:
                return text.strip().strip('"')
        except LLMError:
            pass
    return f"{previous} {question}"


def format_history(history: list[dict] | None) -> str:
    turns = _last_turns(history)
    if not turns:
        return ""
    lines = [f"{m['role'].title()}: {m['content'][:600]}" for m in turns]
    return "Conversation so far (for context only):\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------- pipeline

@dataclass
class Plan:
    """Everything decided before generation: what was retrieved and who answers."""
    question: str
    search_query: str
    docs: list[Document]
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    note: str | None = None
    final_answer: str | None = None  # set when no LLM call is needed (abstain / extractive)
    abstained: bool = False
    messages: list = field(default_factory=list)

    def meta(self) -> dict:
        return {"sources": _sources(self.docs), "provider": self.provider, "model": self.model,
                "note": self.note, "search_query": self.search_query}


def plan(question: str, provider: str | None = None, model: str | None = None,
         api_key: str | None = None, k: int | None = None, subject: str | None = None,
         history: list[dict] | None = None) -> Plan:
    question = question.strip()

    # If the chosen LLM has no key, answer with the default no-key option instead of failing.
    provider = provider if provider in PROVIDERS else default_provider()
    note = None
    if not PROVIDERS[provider].ready(api_key):
        fallback = default_provider()
        if not PROVIDERS[fallback].ready():
            fallback = "extractive"
        note = (f"No API key for {PROVIDERS[provider].label}, so this answer used "
                f"{PROVIDERS[fallback].label}. Add a key in Settings for a written explanation.")
        provider, model, api_key = fallback, None, None

    search_query = standalone_question(question, history, provider, model, api_key)
    retriever = get_retriever(k, subject=subject)
    docs = retriever.invoke(search_query)
    if not docs or max(d.metadata["score"] for d in docs) < min_score():
        return Plan(question, search_query, docs, final_answer=config.NOT_FOUND, abstained=True)

    if provider == "extractive":
        text = extractive_answer(search_query, docs, retriever.store.embeddings)
        note = note or "Quoted from the course material without an LLM. Choose a model for a written explanation."
        return Plan(question, search_query, docs, provider="extractive", note=note, final_answer=text)

    messages = PROMPT.invoke({"history": format_history(history), "context": format_context(docs),
                              "question": question}).to_messages()
    return Plan(question, search_query, docs, provider=provider, model=resolved_model(provider, model),
                api_key=api_key, note=note, messages=messages)


def answer(question: str, provider: str | None = None, model: str | None = None,
           api_key: str | None = None, k: int | None = None, use_retrieval: bool = True,
           subject: str | None = None, history: list[dict] | None = None) -> dict:
    if not use_retrieval:  # baseline used by the evaluation script
        text, provider_id, model_used = generate(BASELINE_SYSTEM_PROMPT, question.strip(), provider, model, api_key)
        return {"answer": text, "sources": [], "provider": provider_id, "model": model_used, "abstained": False}

    p = plan(question, provider, model, api_key, k, subject, history)
    if p.final_answer is not None:
        return {"answer": p.final_answer, "abstained": p.abstained, **p.meta()}
    text, _, _ = generate(p.messages[0].content, p.messages[1].content, p.provider, p.model, p.api_key)
    text = remove_unsupported_citations(text, p.docs)
    return {"answer": text, "abstained": text.startswith(config.NOT_FOUND), **p.meta()}


def answer_stream(question: str, provider: str | None = None, model: str | None = None,
                  api_key: str | None = None, k: int | None = None, subject: str | None = None,
                  history: list[dict] | None = None) -> Iterator[dict]:
    """Yield events: {"type": "meta"} (sources etc.), {"type": "token"}..., then {"type": "done"}."""
    p = plan(question, provider, model, api_key, k, subject, history)
    yield {"type": "meta", **p.meta()}
    if p.final_answer is not None:
        yield {"type": "done", "answer": p.final_answer, "abstained": p.abstained}
        return
    parts = []
    for piece in stream(p.messages[0].content, p.messages[1].content, p.provider, p.model, p.api_key):
        parts.append(piece)
        yield {"type": "token", "text": piece}
    text = remove_unsupported_citations("".join(parts).strip(), p.docs)
    yield {"type": "done", "answer": text, "abstained": text.startswith(config.NOT_FOUND)}
