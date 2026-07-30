"""
Compare numeric-rating distributions across sources on ICLR.

For generated reviews we read final_review_openreview.json (rating / soundness /
presentation / contribution) and check whether the value is an integer. For
human reviews we read the same fields from dataset.csv.

This highlights:
  - rating compression (Qwen and Llama collapse onto 8)
  - format-compliance failures (Llama stringified ratings, Qwen paragraph-stuffed)
  - how the model score distributions compare to human scores for the same papers.
"""
import csv
import json
import re
from collections import Counter

from _common import (
    MAIN_MODELS, ALL_SOURCES, iter_paper_dirs, load_openreview,
    ICLR_DATASET_CSV, tee_output,
)


FIELDS = ("rating", "soundness", "presentation", "contribution")


def _coerce_int(v):
    if isinstance(v, int):
        return v, True
    if isinstance(v, float) and v.is_integer():
        return int(v), True
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group(0)), False   # recovered, but was not already int
    return None, False


def collect_model_ratings(model: str):
    """Return (integer_counter_per_field, list_of_non_integer_per_field)."""
    dist = {f: Counter() for f in FIELDS}
    nonnumeric = {f: [] for f in FIELDS}
    n = 0
    for paper_dir in iter_paper_dirs("ICLR", model):
        obj = load_openreview(paper_dir)
        if obj is None:
            continue
        n += 1
        content = obj.get("content", {}) or {}
        for field in FIELDS:
            v = (content.get(field) or {}).get("value")
            if isinstance(v, int):
                dist[field][v] += 1
            else:
                nonnumeric[field].append((paper_dir.name, str(v)[:120]))
    return n, dist, nonnumeric


def collect_human_ratings():
    """Every official ICLR review contributes one value per field."""
    dist = {f: Counter() for f in FIELDS}
    n = 0
    if not ICLR_DATASET_CSV.exists():
        return 0, dist
    with open(ICLR_DATASET_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                reviews = json.loads(row.get("reviews", "[]"))
            except Exception:
                continue
            for rev in reviews:
                if "Official_Review" not in " ".join(rev.get("invitations", [])):
                    continue
                content = rev.get("content", {}) or {}
                n += 1
                for field in FIELDS:
                    v = content.get(field)
                    if isinstance(v, dict):
                        v = v.get("value")
                    val, _ = _coerce_int(v)
                    if val is not None:
                        dist[field][val] += 1
    return n, dist


def main():
    for source in ALL_SOURCES:
        print(f"\n=== {source} ICLR ===")
        if source in MAIN_MODELS:
            n, dist, nonnumeric = collect_model_ratings(source)
            print(f"  papers: {n}")
            for field in FIELDS:
                dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist[field].items()))
                print(f"  {field}: integer-dist = {{{dist_str}}}  | non-integer = {len(nonnumeric[field])}")
                for pid, preview in nonnumeric[field][:3]:
                    print(f"    - {pid}: {preview!r}")
                if len(nonnumeric[field]) > 3:
                    print(f"    ... and {len(nonnumeric[field]) - 3} more")
        else:
            n, dist = collect_human_ratings()
            print(f"  reviews: {n}")
            for field in FIELDS:
                dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist[field].items()))
                print(f"  {field}: integer-dist = {{{dist_str}}}")


if __name__ == "__main__":
    with tee_output(__file__):
        main()
