"""
Count how often ICLR reviews use generic critique boilerplate (phrasings that
could apply to almost any ML paper and therefore carry little signal).

For generated reviews we scan the Q6 (Weaknesses) + Q7 (Suggestions) fields.
For human reviews we scan the weaknesses + questions fields.

Only applied to ICLR because natureCom reviews are a single free-form blob.
"""
import re
from collections import defaultdict

from _common import (
    MAIN_MODELS, ALL_SOURCES, iter_paper_dirs, load_final_review,
    ICLR_DATASET_CSV, _iclr_content_value, tee_output,
)
import csv
import json


GENERIC_PATTERNS = {
    "more <x> analysis/discussion/comparison":
        r"more (?:detailed|comprehensive|thorough) (?:analysis|discussion|comparison)",
    "hyperparameter sensitivity/analysis/study":
        r"hyperparameter (?:sensitivity|analysis|study)",
    "comparison with more/very recent / SOTA work":
        r"comparison (?:to|with) (?:more recent|very recent|state-of-the-art) (?:work|methods?)",
    "real-world application/instances/datasets":
        r"real-world (?:application|instances|datasets?|scenarios?)",
    "more/additional details on":
        r"(?:more|additional) (?:detail|information|details) (?:about|on)",
    "computational cost/complexity/resources":
        r"computational (?:cost|complexity|resources?)",
    "minor revisions":
        r"minor revisions?",
}


def generated_weakness_suggestion_texts(model: str):
    """Yield per-paper concatenated (Q6+Q7) text for a generated-review model."""
    for paper_dir in iter_paper_dirs("ICLR", model):
        data = load_final_review(paper_dir)
        if data is None:
            continue
        qs = data.get("questions", [])
        if len(qs) < 7:
            continue
        yield ((qs[5].get("final_answer") or "") + " "
               + (qs[6].get("final_answer") or ""))


def human_weakness_question_texts():
    """Yield per-review (weaknesses + questions) text for ICLR human reviews."""
    if not ICLR_DATASET_CSV.exists():
        return
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
                text = (_iclr_content_value(content.get("weaknesses")) + " "
                        + _iclr_content_value(content.get("questions")))
                yield text


def main():
    print("ICLR generic critique boilerplate counts in Weaknesses + Suggestions/Questions fields\n")
    for source in ALL_SOURCES:
        counts = defaultdict(int)
        n = 0
        if source in MAIN_MODELS:
            texts = generated_weakness_suggestion_texts(source)
        else:
            texts = human_weakness_question_texts()
        for text in texts:
            n += 1
            low = text.lower()
            for label, pat in GENERIC_PATTERNS.items():
                if re.search(pat, low):
                    counts[label] += 1
        unit = "reviews" if source == "human" else "papers"
        print(f"  {source} (n={n} {unit}):")
        for label in GENERIC_PATTERNS:
            print(f"    {counts[label]:>3}  {label}")
        print()


if __name__ == "__main__":
    with tee_output(__file__):
        main()
