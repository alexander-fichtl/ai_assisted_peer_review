"""
Length distribution of the ICLR Q8 (Ethics) answer per generated-review model.

Rationale: the ICLR ethics field explicitly allows the short answer
"No ethics review needed". GPT and Llama take this short answer in most cases.
Qwen tends to restate the ICLR ethics policy at length even when no review is
needed, inflating its output.

Human reviews do not have a free-text ethics field in the same form (the
openreview schema uses flag_for_ethics_review, which is typically empty), so
this analysis is model-only.
"""
import statistics

from _common import MAIN_MODELS, iter_paper_dirs, load_final_review, tee_output


Q_INDEX = 7  # Q8 is index 7 (0-indexed)


def main():
    print("ICLR Q8 (Ethics) answer length in characters\n")
    print(f"{'model':<8}{'n':>5}{'min':>7}{'p25':>7}{'median':>9}{'p75':>7}{'max':>7}{'mean':>8}")
    for model in MAIN_MODELS:
        lens = []
        for paper_dir in iter_paper_dirs("ICLR", model):
            data = load_final_review(paper_dir)
            if data is None:
                continue
            qs = data.get("questions", [])
            if Q_INDEX >= len(qs):
                continue
            lens.append(len(qs[Q_INDEX].get("final_answer") or ""))
        if not lens:
            continue
        lens.sort()
        p25 = lens[len(lens) // 4]
        med = lens[len(lens) // 2]
        p75 = lens[3 * len(lens) // 4]
        mean = statistics.mean(lens)
        print(f"{model:<8}{len(lens):>5}{lens[0]:>7}{p25:>7}{med:>9}{p75:>7}{lens[-1]:>7}{mean:>8.0f}")


if __name__ == "__main__":
    with tee_output(__file__):
        main()
