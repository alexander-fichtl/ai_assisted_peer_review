"""
Scan reviews for sycophantic / promotional phrases (e.g. "highly recommend",
"suitable for publication", "significant contribution", "well-written").

Reports, per (venue, source):
  - how many reviews contain at least one promotional phrase
  - the per-phrase count
Human reviews are included for comparison.

Rationale: although the prompts do not ask the model for an accept/reject decision,
Llama closes almost every Nature Communications review with overt recommendations.
"""
from collections import defaultdict

from _common import VENUES, ALL_SOURCES, iter_source_reviews, tee_output


PHRASES = [
    "highly recommend",
    "suitable for publication",
    "make a significant impact",
    "potential to make a substantial impact",
    "potential to impact",
    "substantial contribution",
    "timely and relevant",
    "valuable contribution",
    "significant contribution",
    "strongly recommend",
    "well-written",
    "well written",
]


def main():
    for venue in VENUES:
        print(f"\n=== {venue} ===")
        for source in ALL_SOURCES:
            total = 0
            any_hit = 0
            counts = defaultdict(int)
            for _, text in iter_source_reviews(venue, source):
                total += 1
                low = text.lower()
                hit = False
                for ph in PHRASES:
                    if ph in low:
                        counts[ph] += 1
                        hit = True
                if hit:
                    any_hit += 1
            if total == 0:
                continue
            print(f"  {source}: {any_hit}/{total} reviews have >=1 promotional phrase")
            for ph in PHRASES:
                if counts[ph]:
                    print(f"    \"{ph}\": {counts[ph]}")


if __name__ == "__main__":
    with tee_output(__file__):
        main()
