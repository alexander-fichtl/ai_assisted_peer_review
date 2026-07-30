"""
Stage-2 bulk concern matching over both datasets and all three AI sources,
plus the human-vs-human leave-one-out baseline.

Resume-friendly (existing output files are skipped), thread-pooled, with the
same exponential-backoff retry policy as run_full_extraction.py.

Edit the constants below to scale. Set per-dataset budgets to None to process
every ready paper. RUN_HUMAN_BASELINE toggles the leave-one-out calls.
"""

import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

from data_loaders import DATASETS, _count_tokens, load_paper_text, load_reviews
from match_concerns import (
    KIND_AI,
    KIND_H2H,
    MATCH_CHANNELS,
    build_human_union,
    match_review,
    renumber_items,
    save_match,
)


## configuration -- edit these to scale up or down

N_ICLR_PAPERS = None
N_NATURE_PAPERS = None

# compute the human-vs-human leave-one-out baseline. first thing to drop to save
# budget (it roughly doubles the call count), but it is the in-corpus baseline
# for the AI numbers, so keep it on by default.
RUN_HUMAN_BASELINE = True

# paper-text truncation budget fed to the matcher (see truncate_paper_md).
MAX_PAPER_TOKENS = 400000

N_CONCURRENT = 10
MAX_RETRIES = 5
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0

RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


## paths

_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE / "data"
CONCERNS_ROOT = _HERE / "outputs" / "concerns"
OUT_ROOT = _HERE / "outputs" / "matches"

PAPER_BUDGET = {
    "ICLR26_rndm50": N_ICLR_PAPERS,
    "natureCom_nlpterms_31": N_NATURE_PAPERS,
}

AI_SOURCES = ("gpt", "llama", "qwen")

_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


## stage-1 concern loading

def _load_items(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("items", [])


def _concern_files(paper_dir):
    """Return (human_items_by_file, ai_items_by_source) for one paper concern dir."""
    humans = {}
    ais = {}
    for fp in sorted(paper_dir.glob("*.json")):
        stem = fp.stem
        if stem in AI_SOURCES:
            ais[stem] = _load_items(fp)
        elif stem.startswith("human_"):
            humans[fp.name] = _load_items(fp)
    return humans, ais


def _ready_papers(dataset):
    """Papers whose concern dir has all 3 AI files AND every expected human file."""
    ds_concerns = CONCERNS_ROOT / dataset
    if not ds_concerns.exists():
        return []

    # expected human-reviewer count per paper (from the raw review loader).
    humans = load_reviews(dataset, "human", DATA_ROOT)
    expected = Counter(v["paper_id"] for v in humans.values())

    ready = []
    for paper_dir in sorted(p for p in ds_concerns.iterdir() if p.is_dir()):
        pid = paper_dir.name
        human_files, ai_files = _concern_files(paper_dir)
        has_all_ai = all(src in ai_files for src in AI_SOURCES)
        has_all_humans = expected.get(pid, 0) > 0 and len(human_files) == expected[pid]
        if has_all_ai and has_all_humans:
            ready.append(pid)
    return ready


def _suffix_from_human_file(filename):
    # 'human_Cjz9.json' -> 'Cjz9'
    return Path(filename).stem[len("human_"):]


## work-unit construction

def _build_work(dataset, n_papers):
    """Build the full list of match work units for a dataset (after resume-skip)."""
    papers = _ready_papers(dataset)
    available = len(papers)
    if n_papers:
        papers = papers[:n_papers]
    _log(f"  Ready papers (complete Stage-1 set): {available}; processing {len(papers)} "
         f"(budget={n_papers})")

    work = []
    skipped = 0
    for pid in papers:
        paper_dir = CONCERNS_ROOT / dataset / pid
        human_items_by_file, ai_items_by_source = _concern_files(paper_dir)

        # load + truncate the paper text once; reuse across this paper's units.
        try:
            paper_text = load_paper_text(dataset, pid, DATA_ROOT, max_tokens=MAX_PAPER_TOKENS)
        except FileNotFoundError as e:
            _log(f"  [skip {pid}] no paper text: {e}")
            continue
        _log(f"  {pid}: paper ~{_count_tokens(paper_text)} tokens, "
             f"{len(human_items_by_file)} human files, AI={list(ai_items_by_source)}")

        # full unions (all channels): the matcher is only SHOWN critique+strength,
        # but neutral items ride along for Stage 3's with-neutral variant.
        pooled_human = build_human_union(human_items_by_file)

        # AI-vs-human units
        for src in AI_SOURCES:
            out = OUT_ROOT / dataset / pid / f"{src}.json"
            if out.exists():
                skipped += 1
                continue
            ai_items = renumber_items(ai_items_by_source[src], source_file=f"{src}.json")
            work.append({
                "out": out, "paper_text": paper_text,
                "a_items": ai_items, "b_items": pooled_human,
                "dataset": dataset, "paper_id": pid, "source": src,
                "kind": KIND_AI,
            })

        # human-vs-human leave-one-out units
        if RUN_HUMAN_BASELINE and len(human_items_by_file) >= 2:
            for held_file in sorted(human_items_by_file):
                suffix = _suffix_from_human_file(held_file)
                out = OUT_ROOT / dataset / pid / f"h2h_{suffix}.json"
                if out.exists():
                    skipped += 1
                    continue
                a_items = renumber_items(human_items_by_file[held_file],
                                         source_file=held_file, channels=MATCH_CHANNELS)
                others = {f: it for f, it in human_items_by_file.items() if f != held_file}
                b_items = build_human_union(others, MATCH_CHANNELS)
                work.append({
                    "out": out, "paper_text": paper_text,
                    "a_items": a_items, "b_items": b_items,
                    "dataset": dataset, "paper_id": pid, "source": suffix,
                    "kind": KIND_H2H,
                })

    _log(f"  Queued: {len(work)}  Skipped (already done): {skipped}")
    return work


## retry-wrapped matching

def _match_with_retry(unit):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return match_review(
                unit["paper_text"], unit["a_items"], unit["b_items"],
                dataset=unit["dataset"], paper_id=unit["paper_id"],
                source=unit["source"], comparison_kind=unit["kind"],
            )
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == MAX_RETRIES:
                raise
            delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
            delay += random.uniform(0, delay * 0.3)
            _log(f"  [retry {attempt + 1}/{MAX_RETRIES}] {unit['paper_id']}/{unit['source']}: "
                 f"{type(e).__name__}; sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("retry loop fell through")


## driver

def run_dataset(dataset, n_papers):
    _log(f"\n=== {dataset} ===")
    work = _build_work(dataset, n_papers)

    stats = {"calls": 0, "errors": 0, "tokens": 0, "matches": 0}
    total = len(work)
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        futures = {ex.submit(_match_with_retry, u): u for u in work}
        for fut in as_completed(futures):
            u = futures[fut]
            done += 1
            try:
                res = fut.result()
                save_match(res, u["out"])
                stats["calls"] += 1
                stats["matches"] += len(res.matches)
                stats["tokens"] += res.usage.get("total_tokens", 0)
                kind = "H2H" if u["kind"] == KIND_H2H else "AI "
                _log(f"  [{done}/{total}] {kind} {u['paper_id']}/{u['out'].name} "
                     f"A={len(res.ai_items)} B={len(res.human_items)} "
                     f"matches={len(res.matches)} {res.usage.get('total_tokens', 0)} tok")
            except Exception as e:
                stats["errors"] += 1
                _log(f"  [fail {done}/{total}] {u['paper_id']}/{u['out'].name}: {e!r}")

    stats["elapsed"] = time.time() - t0
    return stats


def main():
    print("bulk concern matching (parallel)")
    print(f"  concerns: {CONCERNS_ROOT}")
    print(f"  output:   {OUT_ROOT}")
    print(f"  budgets:  ICLR={N_ICLR_PAPERS}, natureCom={N_NATURE_PAPERS}")
    print(f"  human LOO: {RUN_HUMAN_BASELINE}")
    print(f"  workers:  {N_CONCURRENT}  paper budget: {MAX_PAPER_TOKENS} tokens")

    grand = {"calls": 0, "errors": 0, "tokens": 0, "matches": 0, "elapsed": 0.0}
    for dataset in DATASETS:
        s = run_dataset(dataset, PAPER_BUDGET[dataset])
        for k in grand:
            grand[k] += s[k]

    print("\n=== total ===")
    print(f"  api calls:    {grand['calls']}")
    print(f"  matches:      {grand['matches']}")
    print(f"  errors:       {grand['errors']}")
    print(f"  total tokens: {grand['tokens']}")
    print(f"  elapsed:      {grand['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
