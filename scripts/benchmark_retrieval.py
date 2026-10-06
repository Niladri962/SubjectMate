"""Measure retrieval quality on eval/retrieval_gold.csv (questions with known source files).

    python scripts/benchmark_retrieval.py
    python scripts/benchmark_retrieval.py --mode dense --mode hybrid --mode hybrid+rerank

Metrics (file-level): hit@1, hit@3, hit@5 and MRR@10.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from subjectmate.retriever import get_retriever  # noqa: E402


def evaluate(mode: str, rows: list[dict], verbose: bool) -> dict:
    retriever = get_retriever(k=10, mode=mode)
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    start = time.perf_counter()
    for row in rows:
        expected = set(row["expected_sources"].split("|"))
        ranked = [d.metadata["source"] for d in retriever.invoke(row["question"])]
        rank = next((i for i, s in enumerate(ranked, start=1) if s in expected), None)
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        rr += 1 / rank if rank else 0
        if verbose and rank != 1:
            print(f"    [{mode}] rank={rank}  {row['question']}  -> got {ranked[:2]}")
    n = len(rows)
    return {"mode": mode, **{f"hit@{k}": v / n for k, v in hits.items()}, "mrr@10": rr / n,
            "ms/query": 1000 * (time.perf_counter() - start) / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=ROOT / "eval" / "retrieval_gold.csv")
    parser.add_argument("--mode", action="append", help="dense | hybrid | hybrid+rerank (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print questions not ranked first")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.gold, encoding="utf-8")))
    modes = args.mode or ["dense", "hybrid", "hybrid+rerank"]
    results = [evaluate(m, rows, args.verbose) for m in modes]
    print(f"\n{len(rows)} questions")
    print(f"{'mode':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR@10':>8}{'ms/q':>8}")
    for r in results:
        print(f"{r['mode']:<16}{r['hit@1']:>8.0%}{r['hit@3']:>8.0%}{r['hit@5']:>8.0%}"
              f"{r['mrr@10']:>8.3f}{r['ms/query']:>8.0f}")


if __name__ == "__main__":
    main()
