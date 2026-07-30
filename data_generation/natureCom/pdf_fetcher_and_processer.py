import sys
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root (two levels above data_generation/natureCom/)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import data_generation.pdf_fetcher_docling as api_docling
#PATH_out_pdfs = REPO_ROOT / "data" / "processed_pdfs" / "natureCom"
#PATH_out_reviews = REPO_ROOT / "data" / "processed_reviews" / "natureCom"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_index_set(index_path: Path, *, set_name: str) -> Dict[str, Any]:
    obj = _read_json(index_path)

    # Backwards-compatible: old single-set format
    # {"paper_ids":[...], "paper_meta": {...}}
    if isinstance(obj, dict) and isinstance(obj.get("paper_ids"), list):
        return {
            "name": set_name,
            "total": int(obj.get("total") or len(obj.get("paper_ids") or [])),
            "paper_ids": obj.get("paper_ids", []),
            "paper_meta": obj.get("paper_meta", {}),
        }

    if not isinstance(obj, dict) or not isinstance(obj.get("sets"), list):
        raise ValueError(f"{index_path} is not a valid index.json (missing top-level 'sets' list).")

    target = str(set_name).strip()
    for s in obj["sets"]:
        if not isinstance(s, dict):
            continue
        if str(s.get("name") or "").strip() == target:
            return s

    raise ValueError(f"{index_path}: set not found: {target}")


def _load_index_ids_and_meta(index_path: Path, *, set_name: str) -> Tuple[List[str], Dict[str, Dict[str, Any]], int]:
    s = _load_index_set(index_path, set_name=set_name)

    paper_ids = s.get("paper_ids")
    if not isinstance(paper_ids, list):
        raise ValueError(f"{index_path}: set '{set_name}' missing 'paper_ids' list.")
    ids = [str(x).strip() for x in paper_ids if str(x).strip()]

    paper_meta = s.get("paper_meta", {})
    if paper_meta is None:
        paper_meta = {}
    if not isinstance(paper_meta, dict):
        raise ValueError(f"{index_path}: set '{set_name}' has non-dict 'paper_meta'.")

    meta_by_id: Dict[str, Dict[str, Any]] = {}
    for k, v in paper_meta.items():
        pid = str(k).strip()
        if not pid:
            continue
        meta_by_id[pid] = v if isinstance(v, dict) else {}

    total = s.get("total")
    if isinstance(total, int):
        n_total = total
    else:
        n_total = len(ids)

    return ids, meta_by_id, n_total

def _docling_convert_to_root(pdf_url: str, doc_id: str, dest_root: Path) -> bool:
    """
    Uses api_docling.pdfToJson and writes directly into:
      dest_root/<doc_id>/
    """
    if not pdf_url:
        return False

    dest_dir = dest_root / doc_id
    dest_json = dest_dir / f"{doc_id}.json"
    if dest_json.exists():
        return False

    _ensure_dir(dest_root)

    api_docling.pdfToJson(pdf_url, paper_id=doc_id, output_base=str(dest_root))

    if not dest_json.exists():
        raise RuntimeError(f"Docling did not produce expected JSON: {dest_json}")

    try:
        api_docling.clean_docling_id(doc_id, root=str(dest_root))
    except Exception:
        pass

    return True

def generate_dataset(
    *,
    index_json: Path,
    set_name: str,
    name: str,
    process_pdfs: bool = True,
    process_reviews: bool = True,
) -> None:
    ids, meta_by_id, total = _load_index_ids_and_meta(index_json, set_name=set_name)

    out_dir = REPO_ROOT / "data" / "final_datasets" / name
    _ensure_dir(out_dir)
    out_csv = out_dir / "dataset.csv"

    processed_pdfs_root = REPO_ROOT / "data" / "processed_pdfs" / "natureCom"
    processed_reviews_root = REPO_ROOT / "data" / "processed_reviews" / "natureCom"
    _ensure_dir(processed_pdfs_root)
    _ensure_dir(processed_reviews_root)

    fieldnames = [
        "paper_id",
        "title",
        "pdf_url",
        "peerreview_url",
        "has_peerreview",
        "paper_docling_json",
        "paper_docling_txt",
        "paper_docling_md",
        "review_docling_json",
        "review_docling_txt",
        "review_docling_md",
    ]

    # Load existing rows (so reruns can update paths after earlier failures)
    rows_by_id: Dict[str, Dict[str, str]] = {}
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = (row.get("paper_id") or "").strip()
                if pid:
                    rows_by_id[pid] = {k: (row.get(k) or "") for k in fieldnames}

    converted_papers = 0
    converted_reviews = 0
    missing_meta = 0
    missing_pdf_url = 0
    missing_pr_url = 0

    for pid in ids:
        meta = meta_by_id.get(pid)
        if meta is None:
            # Should not happen if your index generator creates placeholders.
            print(f"[warn] id missing in paper_meta, skipping: {pid}")
            missing_meta += 1
            continue

        title = str(meta.get("title") or "").strip()
        pdf_url = str(meta.get("url_pdf") or "").strip()
        pr_url = str(meta.get("url_peerreview") or "").strip()

        row = rows_by_id.get(pid) or {k: "" for k in fieldnames}
        row["paper_id"] = pid
        row["title"] = title
        row["pdf_url"] = pdf_url
        row["peerreview_url"] = pr_url
        row["has_peerreview"] = "yes" if pr_url else "no"

        # Process paper PDF
        if process_pdfs:
            if pdf_url:
                paper_dir = processed_pdfs_root / pid
                if not (paper_dir / f"{pid}.json").exists():
                    try:
                        if _docling_convert_to_root(pdf_url, pid, processed_pdfs_root):
                            converted_papers += 1
                    except Exception as e:
                        print(f"[warn] Docling failed for paper PDF {pid}: {e}")
            else:
                missing_pdf_url += 1

        # Process peer review PDF
        if process_reviews:
            if pr_url:
                review_dir = processed_reviews_root / pid
                if not (review_dir / f"{pid}.json").exists():
                    try:
                        if _docling_convert_to_root(pr_url, pid, processed_reviews_root):
                            converted_reviews += 1
                    except Exception as e:
                        print(f"[warn] Docling failed for peer review PDF {pid}: {e}")
            else:
                missing_pr_url += 1

        # Update output paths (relative to repo root)
        paper_dir = processed_pdfs_root / pid
        review_dir = processed_reviews_root / pid

        row["paper_docling_json"] = str((paper_dir / f"{pid}.json").relative_to(REPO_ROOT)) if (paper_dir / f"{pid}.json").exists() else ""
        row["paper_docling_txt"]  = str((paper_dir / f"{pid}.txt").relative_to(REPO_ROOT))  if (paper_dir / f"{pid}.txt").exists()  else ""
        row["paper_docling_md"]   = str((paper_dir / f"{pid}.md").relative_to(REPO_ROOT))   if (paper_dir / f"{pid}.md").exists()   else ""

        row["review_docling_json"] = str((review_dir / f"{pid}.json").relative_to(REPO_ROOT)) if (review_dir / f"{pid}.json").exists() else ""
        row["review_docling_txt"]  = str((review_dir / f"{pid}.txt").relative_to(REPO_ROOT))  if (review_dir / f"{pid}.txt").exists()  else ""
        row["review_docling_md"]   = str((review_dir / f"{pid}.md").relative_to(REPO_ROOT))   if (review_dir / f"{pid}.md").exists()   else ""

        rows_by_id[pid] = row

    # Write CSV in index order
    tmp_csv = out_csv.with_suffix(".tmp.csv")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pid in ids:
            if pid in rows_by_id:
                w.writerow(rows_by_id[pid])

    tmp_csv.replace(out_csv)

    print(
        "[done]\n"
        f"  set: {set_name}\n"
        f"  ids in set: {len(ids)} (total field: {total})\n"
        f"  missing meta entries (paper_meta): {missing_meta}\n"
        f"  missing url_pdf: {missing_pdf_url}\n"
        f"  missing url_peerreview: {missing_pr_url}\n"
        f"  converted paper PDFs: {converted_papers}\n"
        f"  converted peer review PDFs: {converted_reviews}\n"
        f"  wrote: {out_csv}\n"
        "  outputs:\n"
        "    - {PATH_out_pdfs}/<paper_id>/\n"
        "    - {PATH_out_reviews}/<paper_id>/\n"
    )


def main() -> None:
    # Your new index.json (Option 1 schema)
    INDEX_JSON = REPO_ROOT / "data" / "papersets_ids" / "natureCom" / "index.json"

    # Which set inside index.json to run
    SET_NAME = "NatureCom_nlpterms_31"

    # Output dataset folder name under data/final_datasets/<NAME>/
    NAME = "natureCom_nlpterms_31"

    generate_dataset(
        index_json=INDEX_JSON,
        set_name=SET_NAME,
        name=NAME,
        process_pdfs=True,
        process_reviews=True,
    )

main()
