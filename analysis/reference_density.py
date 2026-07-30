"""
Count textual references to Figure / Table / Section / Equation (each followed by a digit)
per review. Reports the per-review average for each (venue, source) pair, where source
is one of gpt / qwen / llama / human.

Rationale: a peer review that cites specific figures, tables, sections, or equations is
more grounded in the manuscript. Models differ by more than an order of magnitude on
this metric, and human reviewers sit between GPT and Qwen on most venues.
"""
import re
from collections import defaultdict

from _common import VENUES, ALL_SOURCES, iter_source_reviews, tee_output


PATTERNS = {
    "Figure":   r"\b[Ff]ig(?:ure|\.)?\s*\d",
    "Table":    r"\b[Tt]able\s*\d",
    "Section":  r"\b[Ss]ec(?:tion|\.)?\s*\d",
    "Equation": r"\b[Ee]q(?:uation|\.)?\s*\d",
}


def main():
    for venue in VENUES:
        print(f"\n=== {venue} ===")
        print(f"{'source':<10}{'n':>5}  " + "  ".join(f"{k:>9}" for k in PATTERNS))
        for source in ALL_SOURCES:
            totals = defaultdict(int)
            n = 0
            for _, text in iter_source_reviews(venue, source):
                n += 1
                for k, pat in PATTERNS.items():
                    totals[k] += len(re.findall(pat, text))
            if n == 0:
                continue
            avg = "  ".join(f"{totals[k] / n:>9.2f}" for k in PATTERNS)
            print(f"{source:<10}{n:>5}  {avg}")


if __name__ == "__main__":
    with tee_output(__file__):
        main()
