"""
Stage-1 bulk concern extraction over both datasets and all four reviewer sources.

Resume-friendly: existing output files are skipped. Requests run in a thread pool
with exponential-backoff retries on rate-limit and transient errors.

Edit the constants below to scale up. Set budgets to None to process all papers.

Output: outputs/concerns/<dataset>/<paper_id>/
human_<suffix>.json one per human reviewer
gpt.json / llama.json / qwen.json one per AI source

To Run: python run_full_extraction.py
"""

import threading
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

from data_loaders import DATASETS, SOURCES, load_reviews
from extract_concerns import extract_concerns, save_extraction


## configuration

N_ICLR_PAPERS = None
N_NATURE_PAPERS = None

N_CONCURRENT = 10

MAX_RETRIES = 5
BACKOFF_BASE = 2.0   # initia wait in seconds. doubles each retry
BACKOFF_CAP = 60.0   # max wait between retries

RETRYABLE_EXCEPTIONS = ( #taken from openai docs
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


## paths

_HERE = Path(__file__).resolve().parent
DATA_ROOT = _HERE / "data"
OUT_ROOT = _HERE / "outputs" / "concerns"

PAPER_BUDGET = {
    "ICLR26_rndm50": N_ICLR_PAPERS,
    "natureCom_nlpterms_31": N_NATURE_PAPERS,
}

AI_SOURCES = tuple(s for s in SOURCES if s != "human")

# used to serialize stdout writes from worker threads
_print_lock = threading.Lock()


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


## output path helpers

def _filename_for(review_id, paper_id, source):
    """Stable per-reviewer filename: <source>.json for AI, human_<suffix>.json for humans."""
    if source != "human":
        return f"{source}.json"
    prefix = f"{paper_id}_human_"
    suffix = review_id[len(prefix):] if review_id.startswith(prefix) else review_id
    return f"human_{suffix}.json"


def _output_path(dataset, paper_id, review_id, source):
    return OUT_ROOT / dataset / paper_id / _filename_for(review_id, paper_id, source)


## retry logic

def _extract_with_retry(text, review_id, paper_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return extract_concerns(text, review_id=review_id, paper_id=paper_id)
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == MAX_RETRIES:
                raise
            delay = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
            delay += random.uniform(0, delay * 0.3)  # jitter
            _log(f"  [retry {attempt + 1}/{MAX_RETRIES}] {review_id}: {type(e).__name__}; sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("retry loop fell through")


## per-dataset extraction

def _common_papers(humans, gens_by_source):
    common = {v["paper_id"] for v in humans.values()}
    for src, gens in gens_by_source.items():
        common &= {v["paper_id"] for v in gens.values()}
    return sorted(common)


def extract_dataset(dataset, n_papers):
    _log(f"\n=== {dataset} ===")
    humans = load_reviews(dataset, "human", DATA_ROOT)
    gens_by_source = {src: load_reviews(dataset, src, DATA_ROOT) for src in AI_SOURCES}

    papers = _common_papers(humans, gens_by_source)
    available = len(papers)
    if n_papers:
        papers = papers[:n_papers]
    _log(f"  papers with all sources: {available}, processing: {len(papers)} (budget={n_papers})")

    #build full work list in deterministic order
    work = []  # (out_path, pid, source, review_id, text)
    skipped = 0
    for pid in papers:
        for review_id, rev in humans.items():
            if rev["paper_id"] == pid:
                out = _output_path(dataset, pid, review_id, "human")
                if out.exists():
                    skipped += 1
                else:
                    work.append((out, pid, "human", review_id, rev["text"]))
        for src in AI_SOURCES:
            for review_id, rev in gens_by_source[src].items():
                if rev["paper_id"] == pid:
                    out = _output_path(dataset, pid, review_id, src)
                    if out.exists():
                        skipped += 1
                    else:
                        work.append((out, pid, src, review_id, rev["text"]))

    _log(f"  queued: {len(work)}, skipped: {skipped}, workers: {N_CONCURRENT}")

    stats = {"calls": 0, "skipped": skipped, "errors": 0, "tokens": 0}
    t0 = time.time()
    done_count = 0
    total = len(work)

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
        futures = {
            ex.submit(_extract_with_retry, text, review_id, pid): (out, pid, source, review_id)
            for (out, pid, source, review_id, text) in work
        }
        for fut in as_completed(futures):
            out, pid, source, review_id = futures[fut]
            done_count += 1
            try:
                res = fut.result()
                save_extraction(res, out)
                stats["calls"] += 1
                stats["tokens"] += res.usage.get("total_tokens", 0)
                _log(f"  [{done_count}/{total}] {pid}/{out.name} - {len(res.items)} items, {res.usage.get('total_tokens', 0)} tokens")
            except Exception as e:
                stats["errors"] += 1
                _log(f"  [FAIL {done_count}/{total}] {pid}/{out.name}: {e!r}")

    stats["elapsed"] = time.time() - t0
    return stats


def main():
    print("bulk concern extraction (parallel)")
    print(f"  data root: {DATA_ROOT}")
    print(f"  output:    {OUT_ROOT}")
    print(f"  budgets:   ICLR={N_ICLR_PAPERS}, natureCom={N_NATURE_PAPERS}")
    print(f"  workers:   {N_CONCURRENT}  (max retries: {MAX_RETRIES})")

    grand = {"calls": 0, "skipped": 0, "errors": 0, "tokens": 0, "elapsed": 0.0}
    for dataset in DATASETS:
        s = extract_dataset(dataset, PAPER_BUDGET[dataset])
        for k in grand:
            grand[k] += s[k]

    print("\n=== total ===")
    print(f"  api calls:    {grand['calls']}")
    print(f"  skipped:      {grand['skipped']}")
    print(f"  errors:       {grand['errors']}")
    print(f"  total tokens: {grand['tokens']}")
    print(f"  elapsed:      {grand['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
