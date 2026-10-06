"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# sentence-transformers only needs PyTorch; stop `transformers` from importing TensorFlow,
# which breaks on some machines with an incompatible TensorFlow install.
os.environ.setdefault("USE_TF", "0")

try:  # .env is optional and only used for local development
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

INDEX_DIR = Path(os.getenv("SUBJECTMATE_INDEX_DIR", ROOT / "index"))
PDF_DIR = Path(os.getenv("SUBJECTMATE_PDF_DIR", ROOT / "data" / "pdfs"))

# Must be the same model at index time and query time (stored in index/meta.json).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# "auto" | "local" (sentence-transformers) | "hf-api" (Hugging Face Inference API)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
# If no retrieved chunk reaches this cosine similarity, the system abstains without calling the LLM.
# Unset = use the threshold calibrated for the index's embedding model (stored in index/meta.json).
MIN_SCORE = float(os.environ["MIN_SCORE"]) if os.getenv("MIN_SCORE") else None

# dense | hybrid | hybrid+rerank. "dense" scored best on eval/retrieval_gold.csv
# (run scripts/benchmark_retrieval.py to compare on your own questions).
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "dense")
# Weight of BM25 keyword ranks relative to dense ranks in Reciprocal Rank Fusion (hybrid modes).
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.15"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")  # "none" disables

NOT_FOUND = "I couldn't find enough information about this in the course material."
