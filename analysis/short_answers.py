"""
Find ICLR questions whose final_answer is too short to contain any justification
(threshold: < 30 characters). These are cases where a model returned only
"Rating: N" or an equivalent stub.

Especially relevant for Q2 (Soundness), Q3 (Presentation), Q4 (Contribution),
and Q9 (Overall), which ask the model to reason before giving a numeric score.

Only applies to generated reviews (the ICLR multi-step format). Human reviews
do not have a direct equivalent for these fields.
"""
from _common import MAIN_MODELS, iter_paper_dirs, load_final_review, tee_output


THRESHOLD = 30
TRACKED_QUESTIONS = [2, 3, 4, 9]  # 1-indexed


def main():
    print(f"ICLR final_answer shorter than {THRESHOLD} chars (likely rating-only / stub)\n")
    for model in MAIN_MODELS:
        print(f"=== {model} ===")
        short_by_q = {q: [] for q in TRACKED_QUESTIONS}
        n = 0
        for paper_dir in iter_paper_dirs("ICLR", model):
            data = load_final_review(paper_dir)
            if data is None:
                continue
            qs = data.get("questions", [])
            n += 1
            for q in TRACKED_QUESTIONS:
                if q - 1 >= len(qs):
                    continue
                ans = qs[q - 1].get("final_answer") or ""
                if len(ans) < THRESHOLD:
                    short_by_q[q].append((paper_dir.name, ans))
        for q in TRACKED_QUESTIONS:
            hits = short_by_q[q]
            print(f"  Q{q}: {len(hits)}/{n} rating-only answers")
            for pid, ans in hits[:5]:
                print(f"    - {pid}: {ans!r}")
            if len(hits) > 5:
                print(f"    ... and {len(hits) - 5} more")
        print()


if __name__ == "__main__":
    with tee_output(__file__):
        main()
