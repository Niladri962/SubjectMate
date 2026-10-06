![Uploading Screenshot 2026-10-06 092917.png…]()
# SubjectMate

A course-specific question answering system using Retrieval-Augmented Generation (RAG).
Students ask questions; SubjectMate retrieves the relevant passages from a fixed set of
course files (PDF lecture notes, PowerPoint slides, Word documents, notebooks) and answers
**only from those passages**, citing the file name and page or slide. If the material doesn't
cover a question, it says so instead of guessing.

Team: Chirag Parida, Niladri Ghosh, Sampad Kar

## How it works

```
data/pdfs/ (PDF, PPTX, DOCX, IPYNB, MD, or zips) ──► scripts/build_index.py (run locally)
     unzip (nested too) · skip duplicates · text per page / slide / part + file, subject, page metadata
     LangChain RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
     contextual header "subject | file | page" + chunk ──► all-MiniLM-L6-v2 embeddings
     FAISS index ──► index/  (committed and deployed)

question ──► [subject filter] ──► FAISS cosine search ──► drop duplicate passages ──► top-k
         ──► best score < 0.30 ? ──► "not found" (no LLM call)
         ──► grounded prompt ──► LLM ──► answer with [file, p. N] / [file, slide N] citations
```

## Features

- **Grounded answers with citations.** Every claim cites a file and page or slide. Citations that
  don't match a retrieved passage are removed automatically, because LLMs sometimes invent sources.
- **Says "not found"** when the course material doesn't cover a question, without calling the LLM.
- **Streaming.** The answer appears as it is written, with a Stop button.
- **Follow-up questions.** "What is its time complexity?" after a Dijkstra question is turned into
  a standalone search query (rewritten by the LLM for hosted models, or combined with the previous
  question offline). The last 3 turns are given to the model as context.
- **Any LLM, or none.** Claude, OpenAI, Groq, Gemini, local Ollama, or a no-LLM mode that quotes
  the most relevant sentences. Users can bring their own API key.
- **Math rendering** (KaTeX) for formulas in answers.
- **Saved chats** in the browser, with New chat and Copy buttons.
- **Rate limiting** (20 questions per minute per IP by default, `RATE_LIMIT_PER_MINUTE`) to
  protect server API keys.
- **Tested.** `python -m pytest tests` runs 26 unit and end-to-end API tests.

| Part | File |
|---|---|
| Index builder | [scripts/build_index.py](scripts/build_index.py) |
| Embeddings (local or Hugging Face Inference API) | [subjectmate/embeddings.py](subjectmate/embeddings.py) |
| FAISS retriever (LangChain `BaseRetriever`) | [subjectmate/retriever.py](subjectmate/retriever.py) |
| Grounded prompt and pipeline | [subjectmate/rag.py](subjectmate/rag.py) |
| LLM providers (Claude, OpenAI, Groq, Gemini) | [subjectmate/llm.py](subjectmate/llm.py) |
| API (Vercel serverless function) | [api/index.py](api/index.py) |
| Web UI | [public/](public/) |
| Streamlit UI (local) | [streamlit_app.py](streamlit_app.py) |
| Evaluation | [scripts/evaluate.py](scripts/evaluate.py), [eval/questions.csv](eval/questions.csv) |
| Retrieval benchmark | [scripts/benchmark_retrieval.py](scripts/benchmark_retrieval.py), [eval/retrieval_gold.csv](eval/retrieval_gold.csv) |
| Tests | [tests/](tests/) |

## Retrieval quality

Retrieval is measured on [eval/retrieval_gold.csv](eval/retrieval_gold.csv): 53 questions (33
full sentences, 20 short keyword queries) with the file that answers each one. Run
`python scripts/benchmark_retrieval.py -v` to reproduce.

| Configuration | hit@1 | hit@3 | MRR@10 | latency |
|---|---|---|---|---|
| MiniLM, plain chunks (first version) | 94% | 98% | 0.967 | ~20 ms |
| **MiniLM + contextual chunk headers (default)** | **96%** | **100%** | **0.978** | ~20 ms |
| + hybrid BM25 keyword search (RRF, weight 0.15) | 94% | 100% | 0.972 | ~20 ms |
| + cross-encoder rerank (ms-marco-MiniLM-L-6) | 91% | 96% | 0.943 | ~1.3 s |
| + cross-encoder rerank (bge-reranker-base) | 91% | 98% | 0.945 | ~11 s |
| bge-small-en-v1.5 + headers | 91% | 98% | 0.945 | ~40 ms |

The default is the best-scoring setup. Hybrid search and reranking stay available
(`RETRIEVAL_MODE=hybrid` or `hybrid+rerank`) because they may help on other material. On this
course they tended to rank question banks and quizzes above the lectures. The "not found"
threshold of 0.30 was calibrated so that every course question passes, while 13 of 15
off-topic questions are refused before the LLM is called. The grounded prompt handles the rest.

The API also accepts an optional `subject` in `POST /api/ask` to search a single course (the
web UI searches all subjects).

**Switching LLMs:** the model menu in the top bar switches between Claude, OpenAI, Groq and
Gemini. Providers with a key set on the server work right away. For the others, a user can paste
their own API key in Settings (⚙). That key is kept in their browser and sent only with their
own questions. The model name for each provider can be changed there too.

**Why the deployed app isn't Streamlit:** Vercel runs serverless functions, which can't host a
Streamlit server, and PyTorch is too large for a Vercel function. So the PDFs are indexed
locally with sentence-transformers. On Vercel, a small FastAPI function embeds each question
with the **same model** through the Hugging Face Inference API and searches the prebuilt FAISS
index. The Streamlit UI is still included for running locally.

## 1. Set up locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env          # then fill in the keys you have
```

## 2. Add your course PDFs and build the index

Put the PDFs in `data/pdfs/` (subfolders are fine) and run:

```bash
python scripts/build_index.py
```

This writes `index/index.faiss`, `index/chunks.json` and `index/meta.json`. Run it again
whenever the PDFs change. Scanned PDFs have no text layer and need OCR first (the script
reports any such PDFs).

## 3. Run locally

```bash
uvicorn api.index:app --reload        # web UI at http://localhost:8000
streamlit run streamlit_app.py        # or the Streamlit UI
```

## 4. Deploy to Vercel

1. Push the project to GitHub, **including the `index/` folder**. The PDFs themselves are not
   needed at runtime and are git-ignored by default.
2. On [vercel.com](https://vercel.com/new), import the repository. Keep Framework Preset =
   **Other** and leave the build settings empty.
3. Under **Settings → Environment Variables**, add:
   - `HF_TOKEN`: a free Hugging Face read token ([create one](https://huggingface.co/settings/tokens)). This is required.
   - At least one LLM key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` or `GEMINI_API_KEY`.
   - Optional: `LLM_PROVIDER` (the provider selected by default) and the model overrides in `.env.example`.
4. Deploy. After changing the PDFs, rebuild the index, commit `index/` and push again.

Or deploy from the command line with `npm i -g vercel`, then `vercel` and `vercel --prod`.

API endpoints: `GET /api/health`, `GET /api/documents`, `GET /api/providers`,
`POST /api/ask`, and `POST /api/ask/stream`. The stream endpoint returns newline-delimited JSON
events: `meta` (sources), then `token`... and finally `done` (or `error`). The request body is
`{"question": "...", "provider": "groq", "model": null, "api_key": null, "history": [{"role": "user", "content": "..."}]}`.

## 5. Evaluate

Fill [eval/questions.csv](eval/questions.csv) with your 30 questions. Include unanswerable ones
(`answerable = no`). For answerable questions, add the expected PDF name and page(s). Then run:

```bash
python scripts/evaluate.py --provider groq
```

This runs every question with RAG and without retrieval (the baseline), and reports retrieval
hit@k, abstention accuracy and the share of answers with citations. To score answer correctness
and citation support, grade the `correct` and `citation_supported` columns in
`eval/results.csv` by hand (1/0), then run `python scripts/evaluate.py --summarize`.

## Configuration

All settings are environment variables (see [subjectmate/config.py](subjectmate/config.py)):
`TOP_K` (default 5), `MIN_SCORE` (calibrated per model; 0.30 for MiniLM), `CHUNK_SIZE` (1000),
`CHUNK_OVERLAP` (150), `RETRIEVAL_MODE` (`dense` / `hybrid` / `hybrid+rerank`), `BM25_WEIGHT`
(0.15), `RERANK_MODEL`, `EMBEDDING_MODEL`, and `EMBEDDING_BACKEND` (`auto` / `local` / `hf-api`).

Answers without an API key: if a local [Ollama](https://ollama.com) is running, "Ollama (local,
free)" writes the answers (model `OLLAMA_MODEL`, default `llama3.2:3b`). "No LLM (extract from
notes)" quotes the most relevant sentences with citations and works everywhere, including on
Vercel. If a user picks a provider that has no key, the app falls back to one of these and says so.

When Claude is selected, the default model is `claude-opus-5-5` with `low` effort, which keeps
latency short for grounded answers. It also uses Anthropic's server-side refusal fallback
(`fallbacks: "default"`), so a request declined by a safety classifier is retried on
Anthropic's recommended fallback model.
