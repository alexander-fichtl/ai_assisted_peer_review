"""
Detect reviews that collapse into runs of short, praise-only sentences.

Heuristic:
  - split the review into sentences on '.', '!', '?'
  - a sentence is a "short-praise" sentence if it has 3-15 words, mentions at
    least one praise cue (or starts with a stock subject like "The study" /
    "The manuscript" / "The authors" / "I highly recommend"), and does NOT
    contain a hedge/critique word such as "however", "lacks", "missing",
    "limitation", "concern"
  - a review is flagged if it contains a RUN of >= MIN_RUN consecutive
    short-praise sentences anywhere in the text

Reports, per (venue, source):
  - number of reviews flagged (with at least one run >= MIN_RUN)
  - longest run observed
  - list of flagged paper IDs with their longest run length
"""
import re
from collections import defaultdict

from _common import VENUES, ALL_SOURCES, iter_source_reviews, tee_output


MIN_RUN = 4       # how many consecutive short-praise sentences to flag
MIN_WORDS = 3
MAX_WORDS = 15

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
PRAISE_STARTERS = re.compile(
    r"^\s*(?:the (?:study|manuscript|authors|work|paper|research|findings|results|"
    r"approach|method|proposed|data|analysis|framework|model|technique)|"
    r"this (?:study|manuscript|work|paper|approach|method)|"
    r"their (?:approach|method|work|findings|results)|"
    r"i (?:highly|strongly)? ?recommend|i believe|overall[,]?)\b",
    re.IGNORECASE,
)
PRAISE_WORDS = re.compile(
    r"\b(?:recommend|suitable|significant|valuable|well[- ]written|well[- ]designed|"
    r"well[- ]organized|well[- ]executed|well[- ]structured|well[- ]presented|"
    r"well[- ]reported|compelling|robust|innovative|impact(?:ful)?|contribution|"
    r"strength|excellent|outstanding|thorough|comprehensive|effective|"
    r"high[- ]quality|important|timely|insightful|novel|rigorous|clear)\b",
    re.IGNORECASE,
)
HEDGE_WORDS = re.compile(
    r"\b(?:but|however|although|though|nevertheless|yet|unfortunately|"
    r"lacks?|missing|weak|insufficient|limit(?:ed|ation|ations)|concern|"
    r"issue|problem|error|flaw|inadequate|unclear|confusing|questionable|"
    r"should|would benefit|could benefit|needs?)\b",
    re.IGNORECASE,
)


def is_short_praise(sent: str) -> bool:
    words = sent.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if HEDGE_WORDS.search(sent):
        return False
    return bool(PRAISE_STARTERS.search(sent) or PRAISE_WORDS.search(sent))


def longest_praise_run(text: str):
    sents = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    best = cur = 0
    best_start = cur_start = 0
    for i, s in enumerate(sents):
        if is_short_praise(s):
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best = cur
                best_start = cur_start
        else:
            cur = 0
    return best, sents, best_start


def main():
    print(f"Degenerate-praise detector  (run length >= {MIN_RUN}, "
          f"{MIN_WORDS}-{MAX_WORDS} words per sentence, no hedge/critique words)\n")
    for venue in VENUES:
        print(f"=== {venue} ===")
        for source in ALL_SOURCES:
            flagged = []
            longest_overall = 0
            n = 0
            for pid, text in iter_source_reviews(venue, source):
                n += 1
                best, _, _ = longest_praise_run(text)
                longest_overall = max(longest_overall, best)
                if best >= MIN_RUN:
                    flagged.append((pid, best))
            if n == 0:
                continue
            print(f"  {source}: {len(flagged)}/{n} reviews flagged, "
                  f"longest run observed = {longest_overall}")
            for pid, run_len in sorted(flagged, key=lambda x: -x[1])[:10]:
                print(f"    - {pid}  (longest run = {run_len})")
            if len(flagged) > 10:
                print(f"    ... and {len(flagged) - 10} more")
        print()


def dump_example(venue: str, source: str, pid_prefix: str):
    """Print the longest praise run for one paper (debug helper)."""
    for pid, text in iter_source_reviews(venue, source):
        if not pid.startswith(pid_prefix):
            continue
        best, sents, start = longest_praise_run(text)
        print(f"--- {venue}/{source}/{pid}: run length = {best} ---")
        for i in range(start, min(start + best, len(sents))):
            print(f"  [{i}] {sents[i]}")
        return


if __name__ == "__main__":
    with tee_output(__file__):
        main()
