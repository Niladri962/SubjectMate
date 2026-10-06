"""Build the SubjectMate search index from the course files.

Put the course material in data/pdfs/ (PDF, PPTX, DOCX, IPYNB, MD, TXT, or zips of these).
Run this on your machine whenever the material changes, then commit the index/ folder and redeploy:

    python scripts/build_index.py
    python scripts/build_index.py --pdf-dir path/to/files --chunk-size 800
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subjectmate import config  # noqa: E402  (first: sets environment before ML imports)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from loaders import SUPPORTED, extract_zips, load_file  # noqa: E402
from subjectmate.embeddings import get_embeddings  # noqa: E402
from subjectmate.retriever import CHUNKS_FILE, INDEX_FILE, META_FILE, chunk_header  # noqa: E402

# Instruction some retrieval models expect in front of queries (not documents).
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
# Abstention threshold (best cosine similarity below which the app says "not found"), calibrated
# per model on course vs off-topic questions. Recalibrate if you change the embedding model.
MIN_SCORES = {
    "sentence-transformers/all-MiniLM-L6-v2": 0.30,
}


def find_files(roots: list[Path]) -> list[tuple[Path, Path]]:
    """(file, root) pairs for every supported file, skipping exact duplicates."""
    seen, found = set(), []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUPPORTED or path.name.startswith("._") or not path.is_file():
                continue
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                found.append((path, root))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", type=Path, default=config.PDF_DIR)
    parser.add_argument("--out", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--backend", choices=["auto", "local", "hf-api"], default=config.EMBEDDING_BACKEND)
    args = parser.parse_args()

    # 0. Unpack zips (including nested ones) next to the source folder.
    extracted = args.pdf_dir.parent / "extracted"
    extract_zips(args.pdf_dir, extracted)
    files = find_files([args.pdf_dir, extracted] if extracted.exists() else [args.pdf_dir])
    if not files:
        sys.exit(f"No course files found in {args.pdf_dir}. Add PDFs (or zips of them) and run again.")

    # 1. Load each file page by page (slides for PowerPoint), keeping file name, subject and page.
    pages, documents = [], []
    for path, root in files:
        loaded, total = load_file(path, root)
        if not loaded:
            print(f"  SKIPPED {path.name}: no extractable text (scanned? needs OCR)")
            continue
        pages.extend(loaded)
        subject = loaded[0].metadata["subject"]
        documents.append({"name": path.name, "subject": subject, "pages": total})
        print(f"  [{subject or '-'}] {path.name}: {total} units, {len(loaded)} with text")

    if not pages:
        sys.exit("None of the files contained extractable text.")

    # 2. Split into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(pages)
    counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        key = (c.metadata["subject"], c.metadata["source"])
        counts[key] = counts.get(key, 0) + 1
    for d in documents:
        d["chunks"] = counts.get((d["subject"], d["name"]), 0)
    print(f"Split {len(pages)} pages/slides/parts from {len(files)} files into {len(chunks)} chunks.")

    # 3. Embed each chunk with a contextual header (subject | file | page) so the file and course
    #    context count towards similarity, then store in a FAISS inner-product index
    #    (cosine similarity on normalized vectors).
    print(f"Embedding with {args.model} ({args.backend}) ...")
    embeddings = get_embeddings(args.model, args.backend)
    texts = [chunk_header({"text": c.page_content, **c.metadata}) + "\n" + c.page_content for c in chunks]
    vectors = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    args.out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out / INDEX_FILE))
    (args.out / CHUNKS_FILE).write_text(
        json.dumps([{"text": c.page_content, **c.metadata} for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "embedding_model": args.model,
        "query_instruction": QUERY_INSTRUCTIONS.get(args.model, ""),
        "min_score": MIN_SCORES.get(args.model, 0.3),
        "dimension": int(vectors.shape[1]),
        "num_chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": documents,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Index written to {args.out} ({len(chunks)} chunks from {len(documents)} files).")


if __name__ == "__main__":
    main()
