"""Evaluate SubjectMate on eval/questions.csv, with and without retrieval.

    python scripts/evaluate.py --provider groq
    python scripts/evaluate.py --provider anthropic --no-baseline

Automatic metrics:
  - retrieval hit@k: an expected (source, page) appears among the retrieved passages
  - abstention accuracy: abstains on unanswerable questions and answers the answerable ones
  - citation rate: the answer contains at least one [file.pdf, p. N] citation

The output CSV has empty "correct" and "citation_supported" columns for manual grading
(fill them with 1/0), after which you can rerun with --summarize to compute the final scores.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from subjectmate import rag  # noqa: E402
from subjectmate.llm import LLMError  # noqa: E402

CITATION = re.compile(r"\[[^\[\]]+?\.(?:pdf|pptx|docx|ipynb|md|txt),\s*(?:pp?\.|slides?|parts?)\s*\d", re.IGNORECASE)


def parse_pages(value: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", value or "")}


def retrieval_hit(row: dict, sources: list[dict]) -> int | str:
    expected_source = (row.get("expected_source") or "").strip()
    if not expected_source:
        return ""
    pages = parse_pages(row.get("expected_pages", ""))
    return int(any(
        s["source"] == expected_source and (not pages or s["page"] in pages) for s in sources
    ))


def run(args: argparse.Namespace) -> None:
    rows = list(csv.DictReader(open(args.questions, encoding="utf-8")))
    print(f"Evaluating {len(rows)} questions with provider={args.provider or 'default'}")
    out_rows = []
    for row in rows:
        answerable = row["answerable"].strip().lower() in {"yes", "y", "true", "1"}
        modes = [("rag", True)] + ([] if args.no_baseline else [("no_retrieval", False)])
        for mode, use_retrieval in modes:
            try:
                result = rag.answer(row["question"], provider=args.provider, model=args.model,
                                    k=args.k, use_retrieval=use_retrieval)
                answer, sources, abstained = result["answer"], result["sources"], result["abstained"]
            except LLMError as exc:
                answer, sources, abstained = f"ERROR: {exc}", [], False
            out_rows.append({
                "id": row["id"],
                "mode": mode,
                "question": row["question"],
                "answerable": int(answerable),
                "expected_answer": row.get("expected_answer", ""),
                "answer": answer,
                "retrieved": "; ".join(f"{s['source']} {s['unit']} {s['page']} ({s['score']})" for s in sources),
                "retrieval_hit": retrieval_hit(row, sources) if use_retrieval and answerable else "",
                "abstained": int(abstained),
                "abstention_correct": int(abstained != answerable),
                "has_citation": int(bool(CITATION.search(answer))),
                "correct": "",
                "citation_supported": "",
            })
            print(f"  [{row['id']}] {mode:<12} abstained={int(abstained)}  {answer[:70]!r}")
            time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nResults written to {args.out}")
    summarize(args.out)


def _mean(values: list) -> str:
    nums = [int(v) for v in values if str(v).strip() != ""]
    return f"{sum(nums) / len(nums):.0%} (n={len(nums)})" if nums else "-"


def summarize(path: Path) -> None:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for mode in ("rag", "no_retrieval"):
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        answerable = [r for r in subset if r["answerable"] == "1"]
        print(f"\n== {mode} ==")
        print(f"  retrieval hit@k        {_mean([r['retrieval_hit'] for r in answerable])}")
        print(f"  abstention accuracy    {_mean([r['abstention_correct'] for r in subset])}")
        print(f"  answers with citation  {_mean([r['has_citation'] for r in answerable])}")
        print(f"  correct (manual)       {_mean([r['correct'] for r in subset])}")
        print(f"  citation supported     {_mean([r['citation_supported'] for r in answerable])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results.csv")
    parser.add_argument("--provider", default=None, help="anthropic | openai | groq | gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--no-baseline", action="store_true", help="skip the no-retrieval comparison")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between calls (rate limits)")
    parser.add_argument("--summarize", action="store_true", help="only print scores from --out")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
