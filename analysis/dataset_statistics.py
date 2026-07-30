"""
Dataset and Review Statistics
Computes statistics for ICLR and NatureCom datasets:
  Input papers:
    - Length before formatting (.txt) and after formatting (.md), in words
    - Rejected vs. accepted (ICLR only)
    - Number of images per paper
    - Number of human reviews per paper
  Average generated review length (per model)
  Average human review length
  Evaluation scores:
    - GranuScore per model and human (mean ± stdev)
    - LLM-as-a-judge subscores + overall per source (mean ± stdev)
    - Given score (numeric rating in review) per model and human (mean ± stdev)
"""


#####----- Imports ----------------------------------------------------------------------------------------------------------#####
import json
import csv
import os
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
BASE = Path(__file__).resolve().parent.parent  # project root (one level above analysis/)
DATA = BASE / "data"

OUT_PATH = Path(__file__).resolve().parent / "dataset_statistics.txt"

EXCLUDED_SUFFIXES = ("_NoFormulasOOM", "_OOM", "_NoFormulas")
EXCLUDED_NAMES = {"gpt_chat"}
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Helpers ----------------------------------------------------------------------------------------------------------#####
def count_words(text) -> int:
    """Returns 0 for None or empty; converts non-strings automatically."""
    if not text:
        return 0
    return len(str(text).split())

def summarize(values: list) -> dict | None:
    if not values:
        return None
    return {
        "n":      len(values),
        "mean":   round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "min":    min(values),
        "max":    max(values),
        "stdev":  round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
    }

def print_summary(s: dict | None) -> None:
    if s is None:
        print("  N/A (no data)")
    else:
        print(f"  {s['mean']} +/- {s['stdev']}  (median={s['median']}, min={s['min']}, max={s['max']}, n={s['n']})")

def count_images(artifacts_dir: Path) -> int:
    if not artifacts_dir.exists():
        return 0
    image_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    return sum(1 for f in artifacts_dir.iterdir() if f.suffix.lower() in image_suffixes)

def print_generated_review_stats(gen_root: Path, review_reader) -> None:
    """Print per-model word-count stats for generated reviews under gen_root.
    review_reader(paper_dir: Path) -> int | None  (word count, or None to skip paper)
    """
    print("\n-- Generated Reviews (per model) --")
    for model_dir in sorted(gen_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if any(model.endswith(s) for s in EXCLUDED_SUFFIXES) or model in EXCLUDED_NAMES:
            continue
        word_counts = [
            wc for paper_dir in sorted(model_dir.iterdir())
            if paper_dir.is_dir()
            for wc in [review_reader(paper_dir)]
            if wc is not None
        ]
        print(f"  [{model}]", end="  ")
        print_summary(summarize(word_counts))

def print_generation_errors(gen_root: Path, review_reader) -> None:
    """Print per-model list of paper IDs for which no review was generated (review_reader returned None)."""
    print("\n-- Generation Errors (missing output, per model) --")
    for model_dir in sorted(gen_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if any(model.endswith(s) for s in EXCLUDED_SUFFIXES) or model in EXCLUDED_NAMES:
            continue
        failed_ids = [
            paper_dir.name
            for paper_dir in sorted(model_dir.iterdir())
            if paper_dir.is_dir() and review_reader(paper_dir) is None
        ]
        total = sum(1 for p in model_dir.iterdir() if p.is_dir())
        print(f"  [{model}]  {len(failed_ids)} / {total} failed")
        if failed_ids:
            for pid in failed_ids:
                print(f"    - {pid}")


def print_granuscore_stats(plots_dir: Path) -> None:
    """Print GranuScore mean ± stdev per LLM model and for human reviews."""
    print("\n-- GranuScore (per source) --")
    sources = [
        ("gpt",   plots_dir / "scored_generated_gpt.json"),
        ("llama", plots_dir / "scored_generated_llama.json"),
        ("qwen",  plots_dir / "scored_generated_qwen.json"),
        ("human", plots_dir / "scored_human.json"),
    ]
    for label, path in sources:
        if not path.exists():
            print(f"  [{label}]  N/A (file not found)")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = [
            r["granu_score"]
            for r in data.get("reviews", {}).values()
            if isinstance(r.get("granu_score"), (int, float))
        ]
        print(f"  [{label}]", end="  ")
        print_summary(summarize(scores))


def print_llmjudge_stats(eval_dir: Path) -> None:
    """Print LLM-as-a-judge overall + subscores (mean ± stdev) per source."""
    judgements_dir = eval_dir / "judgements"
    if not judgements_dir.exists():
        print("\n-- LLM-as-a-Judge: N/A (no data) --")
        return

    SUBSCORES = ["understanding", "coverage", "evidence_support", "constructiveness", "conciseness"]
    scores_by_source: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for paper_dir in sorted(judgements_dir.iterdir()):
        if not paper_dir.is_dir():
            continue
        for jf in sorted(paper_dir.glob("*.json")):
            try:
                jdata = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            source = jdata.get("source", "unknown")
            judgement = jdata.get("judgement", {})
            overall = judgement.get("overall_score")
            if isinstance(overall, (int, float)):
                scores_by_source[source]["overall"].append(float(overall))
            for sub in SUBSCORES:
                val = judgement.get("subscores", {}).get(sub)
                if isinstance(val, (int, float)):
                    scores_by_source[source][sub].append(float(val))

    model_name = eval_dir.parent.name  # e.g. "gpt-5"
    print(f"\n-- LLM-as-a-Judge ({model_name}) --")
    for source in sorted(scores_by_source):
        print(f"  [{source}]")
        for dim in ["overall"] + SUBSCORES:
            vals = scores_by_source[source].get(dim, [])
            print(f"    {dim}:", end="  ")
            print_summary(summarize(vals))


def _extract_rating_from_openreview_json(obj) -> float | None:
    """Extract numeric rating from final_review_openreview.json content, matching the
    score-comparison script's _extract_overall_from_json logic."""
    if not isinstance(obj, dict):
        return None
    content = obj.get("content")
    if isinstance(content, dict):
        for field in ("rating", "overall", "recommendation"):
            v = content.get(field)
            if v is None:
                continue
            if isinstance(v, dict):
                v = v.get("value")
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                m = re.search(r"(\d+)", v)
                if m:
                    return float(m.group(1))
    return None


def print_given_score_stats_iclr(csv_path: Path, gen_root: Path) -> None:
    """Print given-score (paper rating) mean ± stdev for human and LLM reviewers (ICLR).
    Mirrors the score-comparison script: uses final_review_openreview.json content.rating.value
    for LLM reviews; uses dataset.csv content.rating for human reviews."""
    print("\n-- Given Score / Paper Rating (ICLR, per source) --")

    # ── Human given scores (from dataset.csv) ──────────────────
    human_scores: list[float] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews_raw = json.loads(row.get("reviews", "[]"))
            for rev in reviews_raw:
                invitations = " ".join(rev.get("invitations", []))
                if "Official_Review" not in invitations:
                    continue
                content = rev.get("content", {})
                for field in ("rating", "overall", "recommendation"):
                    v = content.get(field)
                    if v is None:
                        continue
                    if isinstance(v, dict):
                        v = v.get("value")
                    if isinstance(v, (int, float)):
                        human_scores.append(float(v))
                        break
                    if isinstance(v, str):
                        m = re.search(r"(\d+)", v)
                        if m:
                            human_scores.append(float(m.group(1)))
                            break
    print("  [human]", end="  ")
    print_summary(summarize(human_scores))

    # ── LLM given scores (from final_review_openreview.json) ───
    for model_dir in sorted(gen_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if any(model.endswith(s) for s in EXCLUDED_SUFFIXES) or model in EXCLUDED_NAMES:
            continue
        llm_scores: list[float] = []
        for paper_dir in sorted(model_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            openreview_json = paper_dir / "final_review_openreview.json"
            if not openreview_json.exists():
                continue
            try:
                obj = json.loads(openreview_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            score = _extract_rating_from_openreview_json(obj)
            if score is not None:
                llm_scores.append(score)
        print(f"  [{model}]", end="  ")
        print_summary(summarize(llm_scores))
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Review readers ---------------------------------------------------------------------------------------------------#####
def read_iclr_review(paper_dir: Path) -> int | None:
    """ICLR format: final_review.json with a 'questions' list, or .txt fallback."""
    review_json = paper_dir / "final_review.json"
    if review_json.exists():
        rev = json.loads(review_json.read_text(encoding="utf-8"))
        return sum(
            count_words(q.get("final_answer", ""))
            for q in rev.get("questions", [])
            if isinstance(q.get("final_answer"), str)
        )
    review_txt = paper_dir / "final_review.txt"
    if review_txt.exists():
        return count_words(review_txt.read_text(encoding="utf-8", errors="replace"))
    return None

def read_naturecom_review(paper_dir: Path) -> int | None:
    """NatureCom format: final_review.json or <pid>.json with a 'final_answer' field, or .txt fallback."""
    pid = paper_dir.name
    for review_file in [paper_dir / "final_review.json", paper_dir / f"{pid}.json"]:
        if review_file.exists():
            rev = json.loads(review_file.read_text(encoding="utf-8"))
            fa = rev.get("final_answer", "")
            if isinstance(fa, list):
                return sum(count_words(x) for x in fa)
            return count_words(fa)
    txt_file = paper_dir / f"{pid}.txt"
    if txt_file.exists():
        return count_words(txt_file.read_text(encoding="utf-8", errors="replace"))
    return None
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- ICLR -------------------------------------------------------------------------------------------------------------#####
def collect_iclr_stats():
    print("\n" + "=" * 60)
    print("ICLR DATASET STATISTICS  (ICLR26_rndm50)")
    print("=" * 60)

    csv_path    = DATA / "final_datasets" / "ICLR26_rndm50" / "dataset.csv"
    pdf_root    = DATA / "processed_pdfs" / "ICLR"
    gen_root    = DATA / "reviews_generated" / "ICLR"
    granu_dir   = BASE / "review_evaluation" / "granuscore" / "plots" / "ICLR"
    llmjudge_dir = DATA / "llmjudge_evals" / "gpt-5" / "ICLR26_rndm50"

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ── Paper-level stats ──────────────────────
    words_before, words_after = [], []
    images_per_paper = []
    human_review_counts = []
    human_review_words = []
    status_counts = defaultdict(int)

    for row in rows:
        pid = row["id"]
        status_counts[row["status_bucket"]] += 1

        paper_dir = pdf_root / pid
        txt_file = paper_dir / f"{pid}.txt"
        md_file  = paper_dir / f"{pid}.md"

        if txt_file.exists():
            words_before.append(count_words(txt_file.read_text(encoding="utf-8")))
        if md_file.exists():
            words_after.append(count_words(md_file.read_text(encoding="utf-8")))

        images_per_paper.append(count_images(paper_dir / f"{pid}_artifacts"))

        # Human reviews from dataset.csv (filter out meta-reviews)
        reviews_raw = json.loads(row["reviews"])
        official_reviews = [
            r for r in reviews_raw
            if any("Official_Review" in inv for inv in r.get("invitations", []))
        ]
        human_review_counts.append(len(official_reviews))

        text_fields = ["summary", "strengths", "weaknesses", "questions"]
        for rev in official_reviews:
            content = rev.get("content", {})
            total_words = sum(
                count_words(v.get("value", "") if isinstance(v, dict) else v)
                for field in text_fields
                for v in [content.get(field, {})]
            )
            human_review_words.append(total_words)

    print("\n── Input Papers ──")
    print(f"Total papers: {len(rows)}")
    print("\nStatus distribution:")
    for status, cnt in sorted(status_counts.items()):
        print(f"  {status}: {cnt}")

    print(f"\nPaper length BEFORE formatting (.txt):")
    print_summary(summarize(words_before))

    print(f"Paper length AFTER formatting (.md):")
    print_summary(summarize(words_after))

    print(f"\nImages per paper:")
    print_summary(summarize(images_per_paper))

    print(f"\nHuman reviews per paper (official only):")
    print_summary(summarize(human_review_counts))

    print(f"\nAverage human review length (words, text fields: summary+strengths+weaknesses+questions):")
    print_summary(summarize(human_review_words))

    # ── Generated review stats (per model) ────
    print_generated_review_stats(gen_root, read_iclr_review)
    print_generation_errors(gen_root, read_iclr_review)

    # ── Evaluation scores ──────────────────────
    print_granuscore_stats(granu_dir)
    print_llmjudge_stats(llmjudge_dir)
    print_given_score_stats_iclr(csv_path, gen_root)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- NatureCom --------------------------------------------------------------------------------------------------------#####
def collect_naturecom_stats():
    print("\n" + "=" * 60)
    print("NATURECOM DATASET STATISTICS  (natureCom_nlpterms_31)")
    print("=" * 60)

    csv_path  = DATA / "final_datasets" / "natureCom_nlpterms_31" / "dataset.csv"
    pdf_root  = DATA / "processed_pdfs" / "natureCom"
    gen_root  = DATA / "reviews_generated" / "natureCom"
    human_root = DATA / "reviews_human" / "natureCom"
    granu_dir    = BASE / "review_evaluation" / "granuscore" / "plots" / "natureCom"
    llmjudge_dir = DATA / "llmjudge_evals" / "gpt-5" / "natureCom"

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ── Paper-level stats ──────────────────────
    words_before, words_after = [], []
    images_per_paper = []
    human_review_counts = []
    human_review_words = []

    for row in rows:
        pid = row["paper_id"]

        paper_dir = pdf_root / pid
        txt_file = paper_dir / f"{pid}.txt"
        md_file  = paper_dir / f"{pid}.md"

        if txt_file.exists():
            words_before.append(count_words(txt_file.read_text(encoding="utf-8")))
        if md_file.exists():
            words_after.append(count_words(md_file.read_text(encoding="utf-8")))

        images_per_paper.append(count_images(paper_dir / f"{pid}_artifacts"))

        # Human reviews
        human_file = human_root / f"reviews_{pid}.json"
        if human_file.exists():
            hr = json.loads(human_file.read_text(encoding="utf-8"))
            # Unique reviewers across all remark rounds
            reviewer_ids = {
                entry.get("reviewer_id")
                for key in ("first_remarks", "second_remarks", "third_remarks")
                for entry in hr.get(key, [])
            }
            human_review_counts.append(len(reviewer_ids))
            # Word count per reviewer from their first remarks
            for entry in hr.get("first_remarks", []):
                human_review_words.append(count_words(entry.get("remarks_to_the_author", "")))
        else:
            human_review_counts.append(0)

    print("\n── Input Papers ──")
    print(f"Total papers: {len(rows)}")

    print(f"\nPaper length BEFORE formatting (.txt):")
    print_summary(summarize(words_before))

    print(f"Paper length AFTER formatting (.md):")
    print_summary(summarize(words_after))

    print(f"\nImages per paper:")
    print_summary(summarize(images_per_paper))

    print(f"\nHuman reviewers per paper (unique reviewer_id):")
    print_summary(summarize(human_review_counts))

    print(f"\nHuman review length - first_remarks per reviewer (words):")
    print_summary(summarize(human_review_words))

    # ── Generated review stats (per model) ────
    print_generated_review_stats(gen_root, read_naturecom_review)
    print_generation_errors(gen_root, read_naturecom_review)

    # ── Evaluation scores ──────────────────────
    print_granuscore_stats(granu_dir)
    print_llmjudge_stats(llmjudge_dir)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Main -------------------------------------------------------------------------------------------------------------#####
if __name__ == "__main__":
    class Tee:
        def __init__(self, file): self._file = file
        def write(self, msg):
            self._file.write(msg)
            enc = sys.__stdout__.encoding or "utf-8"
            sys.__stdout__.write(msg.encode(enc, errors="replace").decode(enc))
        def flush(self): self._file.flush(); sys.__stdout__.flush()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        collect_iclr_stats()
        collect_naturecom_stats()
        sys.stdout = sys.__stdout__

    print(f"\nOutput saved to: {OUT_PATH}")
#####------------------------------------------------------------------------------------------------------------------------#####
