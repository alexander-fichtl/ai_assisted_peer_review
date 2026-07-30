"""Shared helpers for the analysis scripts.

Also provides a small Tee context manager so every script can print to stdout *and*
save its output to a companion file in analysis/ (same pattern as dataset_statistics.py).
"""
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # project root
ANALYSIS_DIR = Path(__file__).resolve().parent
DATA = BASE / "data"
GEN_ROOT = DATA / "reviews_generated"
HUMAN_NATURE_ROOT = DATA / "reviews_human" / "natureCom"
ICLR_DATASET_CSV = DATA / "final_datasets" / "ICLR26_rndm50" / "dataset.csv"

EXCLUDED_SUFFIXES = ("_NoFormulasOOM", "_OOM", "_NoFormulas", "_tokenLimit")
EXCLUDED_NAMES = {"gpt_chat"}

VENUES = ("ICLR", "natureCom")
MAIN_MODELS = ("gpt", "qwen", "llama")
ALL_SOURCES = ("gpt", "qwen", "llama", "human")  # includes human reviews for comparison


# ── Generated-review helpers ────────────────────────────────────────────────
def iter_paper_dirs(venue: str, model: str):
    """Yield Path objects for each paper folder under data/reviews_generated/<venue>/<model>."""
    root = GEN_ROOT / venue / model
    if not root.exists():
        return
    for paper_dir in sorted(root.iterdir()):
        if paper_dir.is_dir():
            yield paper_dir


def load_final_review(paper_dir: Path):
    f = paper_dir / "final_review.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_openreview(paper_dir: Path):
    f = paper_dir / "final_review_openreview.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_review_text(data: dict, venue: str) -> str:
    """Return the concatenated review text (all answers for ICLR, single blob for natureCom)."""
    if data is None:
        return ""
    if venue == "ICLR":
        return " ".join((q.get("final_answer") or "") for q in data.get("questions", []))
    return data.get("final_answer") or ""


# ── Human-review helpers ────────────────────────────────────────────────────
def _iclr_content_value(field):
    """ICLR content fields are either strings or {'value': '...'} dicts."""
    if isinstance(field, dict):
        return field.get("value", "") or ""
    return field or ""


def iter_human_reviews(venue: str):
    """Yield (paper_id, review_text) pairs for each individual human review.

    - ICLR: parse dataset.csv, keep only official reviews, concatenate the
      free-text fields (summary + strengths + weaknesses + questions).
    - natureCom: read reviews_<pid>.json and yield each first-round reviewer
      remark separately.
    """
    if venue == "ICLR":
        if not ICLR_DATASET_CSV.exists():
            return
        with open(ICLR_DATASET_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("id", "")
                try:
                    reviews = json.loads(row.get("reviews", "[]"))
                except Exception:
                    continue
                for rev in reviews:
                    inv = " ".join(rev.get("invitations", []))
                    if "Official_Review" not in inv:
                        continue
                    content = rev.get("content", {}) or {}
                    parts = [
                        _iclr_content_value(content.get("summary")),
                        _iclr_content_value(content.get("strengths")),
                        _iclr_content_value(content.get("weaknesses")),
                        _iclr_content_value(content.get("questions")),
                    ]
                    text = "\n\n".join(p for p in parts if p)
                    yield pid, text
    elif venue == "natureCom":
        if not HUMAN_NATURE_ROOT.exists():
            return
        for f in sorted(HUMAN_NATURE_ROOT.glob("reviews_*.json")):
            pid = f.stem.replace("reviews_", "")
            try:
                obj = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for entry in obj.get("first_remarks", []) or []:
                text = entry.get("remarks_to_the_author", "") or ""
                if text:
                    yield pid, text


def iter_source_reviews(venue: str, source: str):
    """Uniform iterator over (paper_id, review_text) for one source.

    - If source in MAIN_MODELS: concatenated generated review per paper.
    - If source == 'human': one entry per individual human review.
    """
    if source == "human":
        yield from iter_human_reviews(venue)
        return
    for paper_dir in iter_paper_dirs(venue, source):
        data = load_final_review(paper_dir)
        if data is None:
            continue
        yield paper_dir.name, get_review_text(data, venue)


# ── Output helper ───────────────────────────────────────────────────────────
class Tee:
    """Write stdout both to the real stdout and to a file (cp1252-safe)."""
    def __init__(self, fh):
        self._fh = fh
    def write(self, msg):
        self._fh.write(msg)
        enc = sys.__stdout__.encoding or "utf-8"
        sys.__stdout__.write(msg.encode(enc, errors="replace").decode(enc))
    def flush(self):
        self._fh.flush()
        sys.__stdout__.flush()


def tee_output(script_file: str):
    """Return a context-manager-like object that redirects stdout to analysis/<name>.txt.

    Usage:
        with tee_output(__file__):
            print("hello")
    """
    out_path = ANALYSIS_DIR / (Path(script_file).stem + ".txt")

    class _Ctx:
        def __enter__(self_inner):
            self_inner._fh = open(out_path, "w", encoding="utf-8")
            self_inner._prev = sys.stdout
            sys.stdout = Tee(self_inner._fh)
            return out_path
        def __exit__(self_inner, exc_type, exc, tb):
            sys.stdout = self_inner._prev
            self_inner._fh.close()
            print(f"\nOutput saved to: {out_path}")
    return _Ctx()
