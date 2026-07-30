###--- Imports ----------------------------------------------------------------------------------------###
import os
import sys
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

#Anchor all paths to the repo root (one level above data_generation/)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
import data_generation.ICLR.openreview_api as api_openreview
import data_generation.pdf_fetcher_docling as api_docling
###----------------------------------------------------------------------------------------------------###



###----- Helper Functions -----------------------------------------------------------------------------###
def _is_meta_review(note: Dict[str, Any]) -> bool:
    c = note.get("content") or {}
    if "metareview" in c:
        return True

    inv = (note.get("invitation") or "").lower()
    invs = " ".join(note.get("invitations") or []).lower()

    meta_tokens = ("meta_review", "metareview", "area_chair")
    if any(t in inv for t in meta_tokens):
        return True
    if any(t in invs for t in meta_tokens):
        return True

    return False

def _openreview_review_to_text(review_obj: Dict[str, Any]) -> str:
    """
    Flatten OpenReview-style review JSON into a readable text blob.
    Same structure for humans and LLM reviews using that schema.
    """
    c = (review_obj or {}).get("content") or {}
    parts: List[str] = []

    def add_block(label: str, key: str) -> None:
        v = c.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v is not None and str(v).strip():
            parts.append(f"{label}:\n{v}")

    add_block("Summary", "summary")
    add_block("Strengths", "strengths")
    add_block("Weaknesses", "weaknesses")
    add_block("Suggestions/Questions", "questions")

    for label, key in [
        ("Soundness", "soundness"),
        ("Presentation", "presentation"),
        ("Contribution", "contribution"),
        ("Overall", "rating"),
        ("Confidence", "confidence"),
        ("Conduct", "code_of_conduct"),
    ]:
        v = c.get(key)
        if isinstance(v, dict):
            v = v.get("value")
        if v is not None and str(v).strip():
            parts.append(f"{label}: {v}")

    v = c.get("flag_for_ethics_review")
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v if str(x).strip())
    if v is not None and str(v).strip():
        parts.append(f"Ethics: {v}")

    if not parts and c:
        parts.append(json.dumps(c, ensure_ascii=False, indent=2))

    return "\n\n".join(parts)

def _extract_human_reviewer_id(review_raw: Dict[str, Any]) -> str:
    """
    Extract stable reviewer id:
    - preferred: .../Reviewer_<id> -> <id>
    - fallback: number, or shortened review id, or hash.
    """
    for key in ("signatures", "writers", "readers"):
        for entry in review_raw.get(key, []) or []:
            if "Reviewer_" in entry:
                rid = entry.split("Reviewer_", 1)[-1]
                rid = "".join(ch for ch in rid if ch.isalnum())
                if rid:
                    return rid

    num = review_raw.get("number")
    if isinstance(num, int):
        return f"r{num}"

    rid = review_raw.get("id")
    if isinstance(rid, str) and rid.strip():
        s = rid.strip()
        if len(s) > 8:
            s = s[-8:]
        s = "".join(ch for ch in s if ch.isalnum())
        if s:
            return s

    digest = hashlib.sha256(json.dumps(review_raw, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"h{digest}"

def _extract_numeric_rating(review_raw_json: str) -> int | None:
    """
    Given review_raw_json (stringified OpenReview-style review),
    return the first integer rating (e.g. 8 from '8: accept, good paper'),
    or None if no rating can be found.
    """
    try:
        obj = json.loads(review_raw_json)
    except Exception:
        # fallback: search digits in the raw string
        m = re.search(r"(\d+)", str(review_raw_json))
        return int(m.group(1)) if m else None

    # Try to find a rating-like field in content
    rating_fields = ["rating", "overall", "recommendation"]  # extend if needed

    if isinstance(obj, dict):
        content = obj.get("content") or obj

        for field in rating_fields:
            if field not in content:
                continue
            rating = content.get(field)
            if isinstance(rating, dict):
                rating = rating.get("value")

            if isinstance(rating, (int, float)):
                return int(rating)
            if isinstance(rating, str):
                m = re.search(r"(\d+)", rating)
                if m:
                    return int(m.group(1))

    # Fallback: if obj itself is a simple rating-like thing
    if isinstance(obj, (int, float)):
        return int(obj)
    if isinstance(obj, str):
        m = re.search(r"(\d+)", obj)
        if m:
            return int(m.group(1))

    return None
###----------------------------------------------------------------------------------------------------###



###----- Functions ------------------------------------------------------------------------------------###
def cleanpapers():
    pids = []
    cleaned_papers = 0
    for pid in pids:
        if api_docling.clean_docling_id(doc_id = pid, root = "data/processed_pdfs/ICLR"):
                cleaned_papers += 1
    print(f"Cleaned (i.e. removed line numbers) {cleaned_papers} papers.")

def generate_dataset_base(amount: int, conference: str, name: str) -> None:
    """
    Build a dataset from a cached OpenReview venue (resumable):
      - Choose <amount> random IDs from data/openreview_basesets/<name>/submissions_jsons/*.json
      - Convert PDFs with Docling into data/processed_pdfs/<id>/...
      - Write data/final_datasets/<name>/{index.json,dataset.csv}
      - Rerunning for when internet connections stops: Reuses index.json, appends only new rows, skip Docling if <id>.json exists.
    """
    base_root = REPO_ROOT / "data" / "openreview_basesets" / (conference.replace("/", "_"))
    json_dir = base_root / "submissions_jsons"
    base_csv_path = base_root / "submissions.csv"

    #If no venue_id dump, dump venue
    if not (base_root.exists() and json_dir.exists() and base_csv_path.exists()):
        REPO_ROOT / "data" / "openreview_basesets".mkdir(parents=True, exist_ok=True)
        api_openreview.dump_venue(venue_id=conference, out_root=REPO_ROOT / "data" / "openreview_basesets")

    ids = sorted(p.stem for p in json_dir.glob("*.json"))

    out_dir = REPO_ROOT / "data" / "final_datasets" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dataset.csv"
    out_idx = out_dir / "index.json"

    #Resume: Reuse chosen IDs if index.json exists, else sample and write it
    if out_idx.exists():
        idx = json.loads(out_idx.read_text(encoding="utf-8"))
        chosen = idx.get("paper_ids") or []
        if not chosen:
            raise RuntimeError(f"{out_idx} exists but has no paper_ids")
    else:
        chosen = sorted(random.sample(ids, amount + 1))
        out_idx.write_text(
            json.dumps(
                {"conference": conference, "name": name, "total": len(chosen), "paper_ids": chosen},
                ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    #Map id -> base CSV row
    id_to_row: Dict[str, Dict[str, str]] = {}
    with base_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id") or ""
            if rid:
                id_to_row[rid] = row

    #CSV header
    if id_to_row:
        base_fields = list(next(iter(id_to_row.values())).keys())
    else:
        base_fields = ["id", "title", "venue", "status_bucket", "forum_url", "pdf_url"]
    remove_fields = {"num_reviews", "num_metareviews"}
    add_fields = ["reviews", "metareview", "llm_review", "other_direct_responses"]
    fieldnames = [c for c in base_fields if c not in remove_fields] + add_fields

    #To continue if programm stops: Do unwritten IDs only
    written_ids = set()
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames and "id" in r.fieldnames:
                for row in r:
                    if row.get("id"):
                        written_ids.add(row["id"])
    else:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    #Append missing rows (with replacement if pdf_url is missing)
    with out_csv.open("a", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)

        #IDs we can use to replace missing-PDF entries
        taken = set(chosen) | written_ids
        available_pool = [x for x in ids if x not in taken]

        i = 0
        while i < len(chosen):
            pid = chosen[i]
            if pid in written_ids:
                i += 1
                continue

            note: Dict[str, Any] = api_openreview.get(json_dir / f"{pid}.json")
            reviews = api_openreview.get_reviews(note)
            metas = api_openreview.get_metareviews(note)
            bucket = api_openreview.get_bucket(note)
            base_row = id_to_row.get(pid, {})
            pdf_url = base_row.get("pdf_url", "")

            #If missing PDF: replace with a new ID and retry
            if not pdf_url:
                print(f"[warn] Missing PDF URL for {pid}")
                if not available_pool:
                    print("[warn] No replacement IDs left; skipping this slot.")
                    #Drop the slot so index/json align
                    chosen.pop(i)
                else:
                    #Get replacement, update the chosen list and index.json
                    new_id = available_pool.pop(random.randrange(len(available_pool)))
                    chosen[i] = new_id
                    out_idx.write_text(
                        json.dumps(
                            {
                                "conference": conference,
                                "name": name,
                                "total": len(chosen),
                                "paper_ids": chosen,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    #retry loop with the new id (do not i+=1)
                continue

            processed_json = Path("data") / "processed_pdfs" / "ICLR" / pid / f"{pid}.json"
            if pdf_url and not processed_json.exists():
                try:
                    api_docling.pdfToJson(pdfpath = pdf_url, paper_id = pid, output_base="data/processed_pdfs/ICLR")
                except Exception as e:
                    print(f"[warn] Docling failed for {pid}: {e}")

            row_out = {k: base_row.get(k, "") for k in fieldnames if k not in add_fields}
            row_out["reviews"] = json.dumps(reviews, ensure_ascii=False)
            row_out["metareview"] = json.dumps(metas[0], ensure_ascii=False) if metas else ""
            row_out["llm_review"] = ""

            others: List[Dict[str, Any]] = []
            for r in ((note.get("details") or {}).get("directReplies") or []):
                inv = (r.get("invitation") or "").lower()
                if not inv and r.get("invitations"):
                    inv = (r["invitations"][0] or "").lower()
                if "decision" in inv:
                    continue
                if "meta_review" in inv or "metareview" in inv or "area_chair_review" in inv:
                    continue
                if "official_review" in inv or inv.endswith("/-/official_review") or "review" in inv:
                    continue
                others.append(r)
            row_out["other_direct_responses"] = json.dumps(others, ensure_ascii=False)

            row_out.setdefault("id", pid)
            if not row_out.get("title"):
                row_out["title"] = api_openreview.get_title(note)
            if not row_out.get("venue"):
                c = note.get("content") or {}
                row_out["venue"] = api_openreview.unwrap(c.get("venue"), "") or api_openreview.unwrap(c.get("venueid"), "") or conference

            writer.writerow(row_out)
            fcsv.flush(); os.fsync(fcsv.fileno()) 
            i += 1

    out_idx.write_text(
        json.dumps(
            {"conference": conference, "name": name, "total": len(chosen), "paper_ids": chosen},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    #Remove line numeration
    cleaned_papers = 0
    for pid in chosen:
        if api_docling.clean_docling_id(doc_id = pid, root = "data/processed_pdfs/ICLR"):
                cleaned_papers += 1

    print(f"[done] Wrote:\n  - {out_csv}\n  - {out_idx}\n[clean] cleaned {cleaned_papers}/{len(chosen)} Docling outputs\nDocling outputs: data/processed_pdfs/<id>/")

def update_dataset_with_llm_reviews(name: str) -> None:
    """
    Second pass: fill the 'llm_review' column in data/final_datasets/<name>/dataset.csv
    using JSONs at data/generated_reviews/<paper_id>/final_review_openreview.json
    for the IDs listed in index.json. Other fields/rows remain unchanged.
    """
    out_dir = REPO_ROOT / "data" / "final_datasets" / name
    idx_path = out_dir / "index.json"
    csv_path = out_dir / "dataset.csv"
    reviews_root = REPO_ROOT / "data" / "generated_reviews"

    if not idx_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing index or dataset CSV under {out_dir}")

    ###--- Read chosen IDs from index.json
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    chosen_ids = idx.get("paper_ids") or []
    if not chosen_ids:
        raise RuntimeError(f"{idx_path} has no 'paper_ids'")

    ###--- Read CSV
    good_rows: List[Dict[str, Any]] = []
    skipped_info: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fr:
        #reader = csv.DictReader(fr)
        #fieldnames = list(reader.fieldnames or [])
        #rows = list(reader)
        reader = csv.DictReader(fr)

        #Normalize header: Drop None/empty, strip whitespace
        raw_fieldnames = list(reader.fieldnames or [])
        fieldnames = [f.strip() for f in raw_fieldnames if f is not None and str(f).strip() != ""]
        if "llm_review" not in fieldnames:
            fieldnames.append("llm_review")

        for line_no, row in enumerate(reader, start=2):  #header is line 1
            # If there are extra cells, DictReader puts them under key None
            if None in row:
                junk = row.get(None) or []
                rid = (row.get("id") or "").strip()
                skipped_info.append(
                    f"line {line_no} (id='{rid}'): {len(junk)} extra cell(s) beyond header"
                )
                continue

            clean_row = {}
            for k, v in row.items():
                if k is None:
                    continue
                kk = k.strip() if isinstance(k, str) else k
                clean_row[kk] = v
            good_rows.append(clean_row)

    ###--- Build lookup from id to row index
    id_to_idx = {}
    for i, row in enumerate(good_rows):
        rid = (row.get("id") or "").strip()
        if rid:
            id_to_idx.setdefault(rid, []).append(i)

    ###--- Update rows with available LLM reviews
    missing_reviews = []
    updated = 0
    for pid in chosen_ids:
        jpath = reviews_root / pid / "final_review_openreview.json"
        if not jpath.exists():
            missing_reviews.append(pid)
            continue
        obj = json.loads(jpath.read_text(encoding="utf-8"))
        payload = json.dumps(obj, ensure_ascii=False)

        for i in id_to_idx.get(pid, []):
            good_rows[i]["llm_review"] = payload
            updated += 1

    tmp_path = csv_path.with_suffix(".tmp.csv")
    with tmp_path.open("w", encoding="utf-8", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in good_rows:
            r.setdefault("llm_review", r.get("llm_review", ""))
            r.pop(None, None) 
            writer.writerow(r)
    tmp_path.replace(csv_path)

    print(f"Updated llm_review for {updated} rows in {csv_path}")
    if missing_reviews:
        print(f"Missing review JSONs for {len(missing_reviews)} IDs: {missing_reviews}")
    if skipped_info:
        print(f"Skipped {len(skipped_info)} malformed CSV row(s):")
        for msg in skipped_info:
            print("  -", msg)

def build_reviews_flat(name: str, model_dirs: List[str]) -> None:
    """
    Normalize all human + model reviews for a dataset into:
        data/final_datasets/<name>/reviews_flat.csv

    One row per review:
      - human_<reviewer_id> for humans
      - <model_name> for models (derived from folder name)

    This function:
      - only reads dataset.csv + generated_reviews_* dirs
    """
    ds_dir = REPO_ROOT / "data" / "final_datasets" / name
    idx_path = ds_dir / "index.json"
    base_csv_path = ds_dir / "dataset.csv"
    out_path = ds_dir / "reviews_flat.csv"

    if not idx_path.exists() or not base_csv_path.exists():
        raise FileNotFoundError(f"Missing index.json or dataset.csv in {ds_dir}")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    paper_ids: List[str] = idx.get("paper_ids") or []
    if not paper_ids:
        raise RuntimeError(f"{idx_path} has no 'paper_ids'")

    # Map id -> row from dataset.csv
    id_to_row: Dict[str, Dict[str, str]] = {}
    with base_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("id") or "").strip()
            if pid:
                id_to_row[pid] = row

    fieldnames = [
        "paper_id",
        "review_id",
        "source_type",
        "source_name",
        "review_raw_json",
        "review_text",
    ]

    rows_out: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    # ---- Human reviews ----
    for pid in paper_ids:
        base_row = id_to_row.get(pid)
        if not base_row:
            continue

        raw = base_row.get("reviews") or "[]"
        try:
            reviews = json.loads(raw)
        except Exception:
            reviews = []

        for r in reviews:
            if not isinstance(r, dict):
                continue
            if _is_meta_review(r):
                continue

            rid = _extract_human_reviewer_id(r)
            source_name = f"human_{rid}"
            review_id = f"{pid}_{source_name}"
            if review_id in seen_ids:
                continue

            review_text = _openreview_review_to_text(r)

            rows_out.append({
                "paper_id": pid,
                "review_id": review_id,
                "source_type": "human",
                "source_name": source_name,
                "review_raw_json": json.dumps(r, ensure_ascii=False),
                "review_text": review_text,
            })
            seen_ids.add(review_id)

    # ---- Model reviews from multiple folders ----
    for model_root in model_dirs:
        model_root_path = (REPO_ROOT / model_root) if not os.path.isabs(model_root) else Path(model_root)
        if not model_root_path.exists():
            print(f"[warn] Model dir not found, skipping: {model_root_path}")
            continue

        base_name = model_root_path.name
        model_name = base_name[len("generated_reviews_"):] if base_name.startswith("generated_reviews_") else base_name

        for pid in paper_ids:
            jpath = model_root_path / pid / "final_review_openreview.json"
            if not jpath.exists():
                continue

            try:
                obj = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[warn] Failed to read model review {jpath}: {e}")
                continue

            if isinstance(obj, dict) and ("content" in obj or "summary" in obj or "Summary" in obj):
                review_text = _openreview_review_to_text(obj)
            elif isinstance(obj, str):
                review_text = obj
            else:
                review_text = json.dumps(obj, ensure_ascii=False)

            source_name = model_name
            review_id = f"{pid}_llm_{model_name}"
            if review_id in seen_ids:
                continue

            rows_out.append({
                "paper_id": pid,
                "review_id": review_id,
                "source_type": "model",
                "source_name": source_name,
                "review_raw_json": json.dumps(obj, ensure_ascii=False),
                "review_text": review_text,
            })
            seen_ids.add(review_id)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"[ok] Wrote normalized reviews to {out_path} with {len(rows_out)} rows")

def build_given_scores(name: str) -> None:
    """
    Build a per-review table of *given scores* for a dataset:

        data/final_datasets/<name>/given_scores.csv

    Uses:
      - dataset.csv  -> to get per-paper decision/status (e.g. status_bucket)
      - reviews_flat.csv -> to get all human + model reviews (and their raw JSON)

    Output columns:
      - paper_id
      - review_id
      - human review   ('yes' / 'no')
      - source         ('human' for humans, model name for models)
      - given score    (numeric rating extracted from review_raw_json)
      - decision       (e.g. status_bucket)
    """
    ds_dir = REPO_ROOT / "data" / "final_datasets" / name
    base_csv_path = ds_dir / "dataset.csv"
    flat_csv_path = ds_dir / "reviews_flat.csv"
    out_path = ds_dir / "given_scores.csv"

    if not base_csv_path.exists() or not flat_csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset.csv or reviews_flat.csv in {ds_dir}. "
            "Run generate_dataset_base(...) and build_reviews_flat(...) first."
        )

    # --- 1) Build paper_id -> decision/status lookup from dataset.csv ---
    decision_lookup: Dict[str, str] = {}
    decision_cols = ["status_bucket", "decision", "final_decision"]

    with base_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("id") or row.get("paper_id") or "").strip()
            if not pid:
                continue

            decision_val = ""
            for col in decision_cols:
                if col in row and row[col] and str(row[col]).strip():
                    decision_val = str(row[col]).strip()
                    break
            decision_lookup[pid] = decision_val

    # --- 2) Read reviews_flat.csv and extract scores per review ---
    with flat_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = ["paper_id", "review_id", "source_type", "source_name", "review_raw_json"]
        for col in required_cols:
            if col not in (reader.fieldnames or []):
                raise KeyError(f"'{col}' column not found in {flat_csv_path}")

        rows_out: List[Dict[str, Any]] = []
        missing_rating = 0
        total_reviews = 0

        for row in reader:
            total_reviews += 1
            pid = (row.get("paper_id") or "").strip()
            if not pid:
                continue
            review_id = (row.get("review_id") or "").strip()
            if not review_id:
                continue

            raw_json = row.get("review_raw_json") or ""
            score = _extract_numeric_rating(raw_json)
            if score is None:
                missing_rating += 1
                continue  # skip reviews without rating

            source_type = (row.get("source_type") or "").strip().lower()
            source_name = (row.get("source_name") or "").strip()

            is_human = (source_type == "human")
            human_flag = "yes" if is_human else "no"
            if is_human:
                source = "human"
            else:
                source = source_name or "llm"

            decision = decision_lookup.get(pid, "")

            rows_out.append({
                "paper_id": pid,
                "review_id": review_id,
                "human review": human_flag,
                "source": source,
                "given score": score,
                "decision": decision,
            })

    # --- 3) Write given_scores.csv ---
    fieldnames = ["paper_id", "review_id", "human review", "source", "given score", "decision"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(
        f"[ok] Wrote given scores to {out_path} with {len(rows_out)} rows "
        f"(skipped {missing_rating} / {total_reviews} reviews without a rating)"
    )
###----------------------------------------------------------------------------------------------------###



###----- Main -----------------------------------------------------------------------------------------###
generate_dataset_base(amount=50, conference="ICLR.cc/2026/Conference", name="ICLR26_rndm50")
#update_dataset_with_llm_reviews(name="ICLR25_rndm50")

"""
build_reviews_flat(
    name="ICLR25_rndm50",
    model_dirs=[
        "data/generated_reviews/gwen3_improvedPrompt_2000cut",
        "data/generated_reviews/llama4_improvedPrompt_512cut",
        "data/generated_reviews/llama4_improvedPrompt_2000cut_2ndIter",
        "data/generated_reviews/llama4_improvedPrompt_fewshot_512cut",
        "data/generated_reviews/llama4_improvedPrompt_fewshot_2000cut",
        "data/generated_reviews/llama4_oldPrompt_512cut"
    ],
)

build_given_scores(name="ICLR25_rndm50")
"""
###----------------------------------------------------------------------------------------------------###

