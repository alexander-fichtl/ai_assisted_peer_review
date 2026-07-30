"""
data loaders for the overlap pipeline

reads from data/ inside this.
two important func
load_human_reviews_iclr(dataset_csv)
    -> dict {review_id: {"paper_id": ..., "text": ..., "raw": <note>}}

load_generated_reviews_iclr(source, gen_root)
-> dict {review_id: {"paper_id": ..., "text": ..., "raw": <dict>}}

Parsing adapted from Alex's/Josefin's sc_code-main/review_evaluation/granuscore/evaluator_granuscore.py
"""

import csv
import json
from pathlib import Path


# key -> (display label, render mode: "block" = prose, "inline" = single score value)
_OR_FIELDS = {
    "summary":("Summary","block"),
    "strengths":("Strengths","block"),
    "weaknesses":("Weaknesses","block"),
    "questions":("Suggestions/Questions",  "block"),
    "soundness":("Soundness","inline"),
    "presentation":("Presentation","inline"),
    "contribution":("Contribution","inline"),
    "rating":("Overall","inline"),
    "confidence":("Confidence","inline"),
    "code_of_conduct":("Conduct","inline"),
    "flag_for_ethics_review": ("Ethics","inline"),
}


## Helper functs
def _unwrap_value(entry):
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _block_text(v):
    if v is None:
        return ""
    if isinstance(v, list):
        items = [str(x).strip() for x in v if str(x).strip()]
        return "\n".join(f"- {x}" for x in items) if items else ""
    return str(v).strip()


def _oneline(v):
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str) and x.strip():
                return x.strip()
    return " ".join(_block_text(v).split())


def render_openreview_content(content):
    if not isinstance(content, dict):
        return ""
    #known keys first in declaration order, then unknown keys alphabetically
    ordered = list(_OR_FIELDS) + sorted(k for k in content if k not in _OR_FIELDS)
    blocks = []
    for k in ordered:
        if k not in content:
            continue
        val = _unwrap_value(content[k])
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        label, mode = _OR_FIELDS.get(k, (k.replace("_", " ").title(), "block"))
        if mode == "inline":
            t = _oneline(val)
            if t:
                blocks.append(f"{label}: {t}")
        else:
            t = _block_text(val)
            if t:
                blocks.append(f"{label}:\n{t}")
    return "\n\n".join(blocks).strip()


## meta-review helpers

_META_TOKENS = ("meta_review", "metareview", "area_chair")


def _is_meta_review(note):
    content = note.get("content") or {}
    if "metareview" in content:
        return True
    invitation = (note.get("invitation") or "").lower()
    invitations = " ".join(note.get("invitations") or []).lower()
    return (
        any(tok in invitation for tok in _META_TOKENS)
        or any(tok in invitations for tok in _META_TOKENS)
    )


def _reviewer_id(note):
    for key in ("signatures", "writers", "readers"):
        for entry in note.get(key, []) or []:
            if "Reviewer_" in entry:
                rid = entry.split("Reviewer_", 1)[-1]
                rid = "".join(c for c in rid if c.isalnum())
                if rid:
                    return rid
    num = note.get("number")
    if isinstance(num, int):
        return f"r{num}"
    nid = note.get("id")
    if isinstance(nid, str) and nid.strip():
        s = "".join(c for c in nid.strip() if c.isalnum())
        return s[-8:] if len(s) > 8 else s
    return "unknown"


def _unique_key(existing, base):
    if base not in existing:
        return base
    n = 2
    while f"{base}_dup{n}" in existing:
        n += 1
    return f"{base}_dup{n}"


def load_human_reviews_iclr(dataset_csv):
    """Load ICLR human reviews from dataset.csv; meta-reviews are excluded."""
    dataset_csv = Path(dataset_csv)
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    out = {}
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = (row.get("id") or "").strip()
            raw_reviews = row.get("reviews") or ""
            if not paper_id or not raw_reviews.strip():
                continue
            notes = json.loads(raw_reviews)
            for note in notes:
                if not isinstance(note, dict) or _is_meta_review(note):
                    continue
                text = render_openreview_content(note.get("content") or {})
                if not text.strip():
                    continue
                rid_short = _reviewer_id(note)
                rid = _unique_key(out, f"{paper_id}_human_{rid_short}")
                out[rid] = {"paper_id": paper_id, "text": text, "raw": note}
    return out


def load_generated_reviews_iclr(source, gen_root, *, json_filename="final_review_openreview.json"):
    """Load ICLR AI-generated reviews. Layout: gen_root/<source>/<paper_id>/<json_filename>."""
    gen_root = Path(gen_root)
    source_dir = gen_root / source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    out = {}
    for paper_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        json_path = paper_dir / json_filename
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        paper_id = str(data.get("pdfname") or paper_dir.name)
        content = data.get("content") or {}
        text = render_openreview_content(content)
        if not text.strip():
            continue
        rid = f"{paper_id}_{source}"
        out[rid] = {"paper_id": paper_id, "text": text, "raw": data}
    return out


# natureCom loaders


def load_human_reviews_naturecom(human_dir, *, paper_id_filter=None):
    """Load natureCom human reviews from reviews_<paper_id>.json files.

    paper_id_filter restricts to papers in dataset.csv; the zip ships 2 extras
    that were removed during curation.
    """
    human_dir = Path(human_dir)
    if not human_dir.exists():
        raise FileNotFoundError(f"Human reviews folder not found: {human_dir}")

    out = {}
    for fp in sorted(human_dir.glob("reviews_*.json")):
        paper_id = fp.stem[len("reviews_"):]
        if paper_id_filter and paper_id not in paper_id_filter:
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        first_remarks = data.get("first_remarks") or []
        if not isinstance(first_remarks, list):
            continue
        for entry in first_remarks:
            if not isinstance(entry, dict):
                continue
            reviewer_id = str(entry.get("reviewer_id", "unknown")).strip() or "unknown"
            text = (entry.get("remarks_to_the_author") or "").strip()
            if not text:
                continue
            rid = _unique_key(out, f"{paper_id}_human_r{reviewer_id}")
            out[rid] = {"paper_id": paper_id, "text": text, "raw": entry}
    return out


def load_generated_reviews_naturecom(source, gen_root, *, json_filename="final_review.json"):
    """Load natureCom AI-generated reviews; body is in the final_answer field."""
    gen_root = Path(gen_root)
    source_dir = gen_root / source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    out = {}
    for paper_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        json_path = paper_dir / json_filename
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        paper_id = str(data.get("pdfname") or paper_dir.name)
        text = (data.get("final_answer") or "").strip()
        if not text:
            continue
        rid = f"{paper_id}_{source}"
        out[rid] = {"paper_id": paper_id, "text": text, "raw": data}
    return out


## paper text loader

_PDF_SUBDIR = {
    "ICLR26_rndm50": "ICLR",
    "natureCom_nlpterms_31": "natureCom",
}

# cut at these back-matter headers to keep token count down.
_TRUNCATE_HEADERS = (
    "references",
    "appendix",
    "acknowledg",
    "supplementary",
    "supplementary material",
    "licenses",
    "license for",
)

_CHARS_PER_TOKEN = 4 


def _count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // _CHARS_PER_TOKEN


def _is_md_header(line):
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False, ""
    title = stripped.lstrip("#").strip()
    #drop leading section numbers: "1 INTRODUCTION", "756 APPENDIX", "A.1 ..."
    tokens = [t for t in title.split() if not _is_numbering_token(t)]
    return True, " ".join(tokens).lower()


def _is_numbering_token(tok):
    """True for tokens like '1', '2.3', 'A.1', '756' (section/line numbers)."""
    core = tok.strip(".)")
    if not core:
        return True
    if all(c.isdigit() or c == "." for c in core):
        return True
    # short alphanumeric section codes: "A.1", "B.2" but not plain words
    return (
        len(core) <= 3
        and core[0].isalpha()
        and all(c.isalnum() or c == "." for c in core)
        and any(c.isdigit() for c in core)
    )


def truncate_paper_md(md, max_tokens=40_000, *, paper_id=""):
    """Cut back-matter headers and hard-cap to max_tokens; strips docling image artifacts."""
    lines = md.splitlines()
    cut_idx = len(lines)
    for i, line in enumerate(lines):
        is_header, title = _is_md_header(line)
        if is_header and any(h in title for h in _TRUNCATE_HEADERS):
            cut_idx = i
            break

    kept = [ln for ln in lines[:cut_idx] if "_artifacts/" not in ln]
    body = "\n".join(kept).strip()

    if _count_tokens(body) > max_tokens:
        budget_chars = max_tokens * _CHARS_PER_TOKEN
        clipped = body[:budget_chars]
        nl = clipped.rfind("\n")
        if nl > budget_chars * 0.8:
            clipped = clipped[:nl]
        body = clipped.rstrip() + "\n\n[... paper text truncated to token budget ...]"

    return body


def load_paper_text(dataset, paper_id, data_root, *, prefer="md", max_chars=120_000, max_tokens=40_000):
    """Return docling paper text for one paper, truncated to budget.

    Reads processed_pdfs/<dataset>/<paper_id>/<paper_id>.md (or .txt).
    Run extract_processed_pdfs.py first if the folder is missing.
    """
    data_root = Path(data_root)
    if dataset not in _PDF_SUBDIR:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {tuple(_PDF_SUBDIR)}")

    paper_dir = data_root / "processed_pdfs" / _PDF_SUBDIR[dataset] / paper_id
    order = (".md", ".txt") if prefer == "md" else (".txt", ".md")

    text = None
    for suffix in order:
        fp = paper_dir / f"{paper_id}{suffix}"
        if fp.exists():
            text = fp.read_text(encoding="utf-8")
            break
    if text is None:
        raise FileNotFoundError(
            f"No paper text for {dataset}/{paper_id} under {paper_dir} "
            f"(looked for {paper_id}.md/.txt). Did you run extract_processed_pdfs.py?"
        )

    if max_tokens is not None:
        text = truncate_paper_md(text, max_tokens=max_tokens, paper_id=paper_id)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[... clipped to char budget ...]"
    return text


DATASETS = ("ICLR26_rndm50", "natureCom_nlpterms_31")
SOURCES = ("human", "gpt", "llama", "qwen")


def load_reviews(dataset, source, data_root):
    data_root = Path(data_root)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}")

    dataset_root = data_root / dataset

    if dataset == "ICLR26_rndm50":
        if source == "human":
            return load_human_reviews_iclr(dataset_root / "dataset.csv")
        return load_generated_reviews_iclr(source, dataset_root / "reviews_generated")

    #natureCom_nlpterms_31
    if source == "human":
        # restrict to the 31 papers in dataset.csv 
        csv_path = dataset_root / "dataset.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            paper_ids = {row["paper_id"] for row in csv.DictReader(f)}
        return load_human_reviews_naturecom(
            dataset_root / "reviews_human", paper_id_filter=paper_ids
        )
    return load_generated_reviews_naturecom(source, dataset_root / "reviews_generated")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data_root = here / "data"
    print(f"Loading from {data_root}\n")

    print("dataset\tsource\tn_reviews\tn_papers\tsample")
    for dataset in DATASETS:
        for source in SOURCES:
            try:
                reviews = load_reviews(dataset, source, data_root)
            except FileNotFoundError:
                print(f"{dataset}\t{source}\tMISSING")
                continue
            n_papers = len({v["paper_id"] for v in reviews.values()})
            sample = next(iter(reviews.values()))["text"][:40].replace("\n", " ") if reviews else ""
            print(f"{dataset}\t{source}\t{len(reviews)}\t{n_papers}\t{sample}")
