"""
Sample papers from Research Square API.

CONCEPT:
- All search params optional.
- Fetch ALL papers via /api/search pagination.
- Build a pool of paper IDs: article_identity (e.g., "rs-106030")
- Sample N random IDs and save them.
- If sample.json exists:
    - keeps existing selected_ids (order preserved)
    - treats None as gaps
    - never re-adds anything in removed_ids
    - fills gaps randomly from remaining pool

NEW STRUCTURE:
1) Create pool file from API  -> create_pool_file(...)
2) Create sample file ONLY from pool file -> create_sample_from_pool_file(...)

USAGE:
  from rs_sampler import create_pool_file, create_sample_from_pool_file

  # 1) Build pool once (API)
  create_pool_file(journal_title="Nature Communications", postedAfter="2024-01-01", postedBefore="2026-01-01")

  # 2) Create/refill sample from pool (no API)
  out = create_sample_from_pool_file(n=33)

  # Optional convenience wrapper (API + sample)
  # out = main(...)

  # Manual pruning:
  # Edit sample.json: set removed entries to null and add them to removed_ids
  {
    "n": 30,
    "selected": [
        {"id":"rs-11253","title":"...","link":"..."},
        {"id":"rs-79354","title":"...","link":"..."},
        null,
        {"id":"rs-46767","title":"...","link":"..."}
        // ...
    ],
    "removed_ids": ["rs-16047"]
    }

  # Run again -> refills gaps (NO API if you call create_sample_from_pool_file)
"""

#####----- Imports ----------------------------------------------------------------------------------------------------------#####
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Setup ------------------------------------------------------------------------------------------------------------#####
API_BASE = "https://www.researchsquare.com/api"
SEARCH_URL = f"{API_BASE}/search"

DEFAULT_PAGE_LIMIT = 100
DEFAULT_TIMEOUT_S = 30

setname = "nlpterms"
samplesize = 33

DEFAULT_POOL_PATH = Path("data", "papersets_ids", "natureCom", "pool_" + setname + ".json")
DEFAULT_SAMPLE_PATH = Path("data", "papersets_ids", "natureCom", "sample_" + setname + "_" + str(samplesize) + ".json")
DEFAULT_INDEX_PATH = Path("data", "papersets_ids", "natureCom", "index.json")
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Helper functions -------------------------------------------------------------------------------------------------#####
def build_candidate_tokens(must_contain_phrases: List[str], must_contain_tokens: List[str]) -> List[str]:
    """
    Tokens we actually query the API with (since API can't do boolean logic).
    - For phrases: query each word separately
    - For tokens: query the token as-is
    """
    candidate_tokens: Set[str] = set()

    for ph in must_contain_phrases:
        for w in normalize_text(ph).split(" "):
            if w:
                candidate_tokens.add(w)

    for tok in must_contain_tokens:
        tok = str(tok).strip()
        if tok:
            candidate_tokens.add(tok)

    return sorted(candidate_tokens)

###--- File IO ----------------------------------------------------------------------------------------------------------------###
def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


###--- Params + Parsing ----------------------------------------------------------------------------------------------------------------###
def compact_params(raw: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Drop None/empty strings so all query params are optional."""
    return {k: v for k, v in raw.items() if v is not None and str(v).strip() != ""}

def parse_search_response(payload: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Expected schema:
    {
      "result": {
        "data": [...],
        "limit": 10,
        "offset": 0,
        "rows": 10,
        "total": 441
      }
    }
    """
    if not isinstance(payload, dict) or "result" not in payload:
        raise RuntimeError("Unexpected response: missing top-level 'result'.")

    result = payload["result"]
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected response: 'result' is not an object.")

    data = result.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response: 'result.data' is not a list.")

    meta = {
        "limit": result.get("limit"),
        "offset": result.get("offset"),
        "rows": result.get("rows"),
        "total": result.get("total"),
    }
    return data, meta

def build_id_pool(items: List[Dict[str, Any]]) -> List[str]:
    """Use article_identity as the paper ID."""
    ids: List[str] = []
    seen: Set[str] = set()

    for it in items:
        pid = it.get("article_identity")
        if isinstance(pid, str) and pid.strip():
            pid = pid.strip()
            if pid not in seen:
                seen.add(pid)
                ids.append(pid)

    return ids

def make_full_url(relative_url: Optional[str]) -> Optional[str]:
    """
    API returns url like: "/article/rs-106030/v1"
    Turn into: "https://www.researchsquare.com/article/rs-106030/v1"
    """
    if not isinstance(relative_url, str) or not relative_url.strip():
        return None
    rel = relative_url.strip()
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    if not rel.startswith("/"):
        rel = "/" + rel
    return "https://www.researchsquare.com" + rel

def build_pool_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build an ID -> metadata map from fetched items.
    ID is article_identity.
    """
    pool: Dict[str, Dict[str, Any]] = {}

    for it in items:
        pid = it.get("article_identity")
        if not isinstance(pid, str) or not pid.strip():
            continue
        pid = pid.strip()

        pool[pid] = {
            "nature_id": "",
            "rs_id": pid,
            "title": it.get("title"),
            "link": make_full_url(it.get("url")),
            "posted_at": it.get("posted_at"),
            "authors": it.get("authors"),
            "journal_title": it.get("journal_title"),
            "subject_areas": it.get("subject_areas"),
            "article_type": it.get("article_type"),
            "status": it.get("status"),
        }

    return pool
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Fetch: Pagination + Offset Cap Workaround ------------------------------------------------------------------------#####
def _fetch_page(
    session: requests.Session,
    params: Dict[str, str],
    limit: int,
    offset: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, str]:
    """
    One request. Returns (items, meta, status_code, text_snippet)
    """
    q = dict(params)
    q["limit"] = str(limit)
    q["offset"] = str(offset)

    resp = session.get(SEARCH_URL, params=q, timeout=DEFAULT_TIMEOUT_S)
    status = resp.status_code
    text_snip = resp.text[:300] if resp.text else ""
    if status != 200:
        return [], {}, status, text_snip

    items, meta = parse_search_response(resp.json())
    return items, meta, status, text_snip

def fetch_all(params: Dict[str, str], page_limit: int, max_pages: Optional[int]) -> Dict[str, Any]:
    """
    Fetch all results using offset+limit pagination.

    IMPORTANT API LIMITATION
    - Research Square rejects offset > 9990.
    - So: if total is large, naive pagination will fail.
    - Workaround: split into smaller queries using postedAfter/postedBefore date windows.

    Behavior:
    1) Try normal pagination.
    2) If offset cap error occurs:
       - require a date window (postedAfter/postedBefore). If missing, we create one by bisecting time.
       - recursively fetch within windows until each window can be paged without exceeding offset cap.

    Result:
    - returns merged unique items (deduped by article_identity)
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    # --- Internal: normal pagination until exhausted OR offset-cap error ---
    def _paginate(p: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        all_items: List[Dict[str, Any]] = []
        offset = 0
        total: Optional[int] = None
        page = 0

        while True:
            page += 1
            if max_pages is not None and page > max_pages:
                break

            items, meta, status, text_snip = _fetch_page(session, p, limit=page_limit, offset=offset)

            if status == 400 and "Offset greater than 9990" in text_snip:
                return all_items, "OFFSET_CAP"

            if status != 200:
                raise RuntimeError(f"HTTP {status}: {text_snip}")

            all_items.extend(items)

            rows = meta.get("rows")
            if total is None and isinstance(meta.get("total"), int):
                total = meta["total"]

            if not rows or int(rows) == 0:
                break

            offset += int(rows)

            if isinstance(total, int) and offset >= total:
                break

        return all_items, None

    # --- Internal: merge + dedupe by article_identity ---
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for it in items:
            pid = it.get("article_identity")
            if isinstance(pid, str) and pid.strip():
                if pid in seen:
                    continue
                seen.add(pid)
            out.append(it)
        return out

    # --- Date splitting strategy ---
    from datetime import datetime

    def _parse_date(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    def _fmt_date(d: datetime) -> str:
        return d.strftime("%Y-%m-%d")

    def _ensure_window(p: Dict[str, str]) -> Dict[str, str]:
        q = dict(p)
        if "postedAfter" not in q or not q["postedAfter"].strip():
            q["postedAfter"] = "2000-01-01"
        if "postedBefore" not in q or not q["postedBefore"].strip():
            q["postedBefore"] = "2100-01-01"
        return q

    def _split_window(p: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        after = _parse_date(p["postedAfter"])
        before = _parse_date(p["postedBefore"])
        if after >= before:
            return dict(p), dict(p)

        mid = after + (before - after) / 2
        mid_date = datetime(mid.year, mid.month, mid.day)

        left = dict(p)
        right = dict(p)

        left["postedBefore"] = _fmt_date(mid_date)
        right["postedAfter"] = _fmt_date(mid_date)

        return left, right

    def _fetch_with_splitting(p: Dict[str, str], depth: int = 0, depth_limit: int = 30) -> List[Dict[str, Any]]:
        if depth > depth_limit:
            raise RuntimeError("Date-splitting depth limit exceeded (query still too large).")

        items, err = _paginate(p)
        if err != "OFFSET_CAP":
            return items

        p = _ensure_window(p)
        left, right = _split_window(p)

        if left == right:
            raise RuntimeError(
                "Cannot split date window further, but API offset cap still hit. "
                "Try adding narrower postedAfter/postedBefore manually."
            )

        return _fetch_with_splitting(left, depth + 1) + _fetch_with_splitting(right, depth + 1)

    merged = _fetch_with_splitting(dict(params))
    merged = _dedupe(merged)

    return {
        "query_params": params,
        "count": len(merged),
        "items": merged,
    }
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Sampling + Refill (Objects, not just IDs) ------------------------------------------------------------------------#####
def normalize_selected(selected: Any, n: int) -> List[Optional[Dict[str, Any]]]:
    """
    Enforce list length exactly n.

    selected entries are either:
      - dict with at least {"rs_id": "..."}  OR
      - None (hole)
    """
    out: List[Optional[Dict[str, Any]]] = []

    if not isinstance(selected, list):
        selected = []

    for x in selected:
        if x is None:
            out.append(None)
        elif isinstance(x, dict) and isinstance(x.get("rs_id"), str) and x["rs_id"].strip():
            out.append(x)
        else:
            out.append(None)

    if len(out) < n:
        out.extend([None] * (n - len(out)))
    elif len(out) > n:
        out = out[:n]

    return out

def sample_unique_ids(pool_ids: List[str], k: int, banned: Set[str]) -> List[str]:
    """Sample k unique IDs from pool_ids, excluding banned."""
    candidates = [x for x in pool_ids if x not in banned]
    if len(candidates) < k:
        raise RuntimeError(f"Not enough candidates: need {k}, have {len(candidates)} after bans.")
    return random.sample(candidates, k)

def refill_sample(sample_obj: Dict[str, Any], pool_map: Dict[str, Dict[str, Any]], n: int) -> Dict[str, Any]:
    """
    selected is a list of objects (or None holes).
    removed_ids bans IDs permanently.
    """
    removed_ids_raw = sample_obj.get("removed_ids", [])
    removed_ids = sorted({str(x).strip() for x in removed_ids_raw if str(x).strip()})
    removed_set = set(removed_ids)

    selected = normalize_selected(sample_obj.get("selected", []), n)

    # Clean selected: removed or duplicates or missing in current pool -> hole
    used: Set[str] = set()
    kept: List[Dict[str, Any]] = []
    for i, obj in enumerate(selected):
        if obj is None:
            continue

        pid = obj.get("rs_id")
        if not isinstance(pid, str) or not pid.strip():
            continue

        pid = pid.strip()

        if pid in removed_set:
            continue

        if pid in used:
            continue

        if pid not in pool_map:
            continue

        kept.append(pool_map[pid])
        used.add(pid)

    missing = n - len(kept)
    new_items: List[Dict[str, Any]] = []

    if missing > 0:
        pool_ids = list(pool_map.keys())
        banned = removed_set | used
        fill_ids = sample_unique_ids(pool_ids, missing, banned=banned)

        for pid in fill_ids:
            new_items.append(pool_map[pid])

    # NEW REQUIREMENT:
    # - newly added items go to the TOP of the list
    # - existing kept items keep relative order, but get pushed down
    new_selected = new_items + kept

    if len(new_items) > 0:
        comment = f"NEW_ITEMS: indices [0..{len(new_items)-1}] | OLD_ITEMS_START_AT: {len(new_items)}"
    else:
        comment = "NEW_ITEMS: none | OLD_ITEMS_START_AT: 0"

    return {
        "n": n,
        "comment": comment,
        "selected": new_selected,
        "removed_ids": removed_ids,
    }

def ban_ids(ids: List[str], sample_path: Path = DEFAULT_SAMPLE_PATH) -> None:
    """
    Mark ids as removed and create holes in selected where they appear.
    """
    if not sample_path.exists():
        write_json(sample_path, {"n": 0, "selected": [], "removed_ids": []})

    sample = read_json(sample_path)

    removed = set(sample.get("removed_ids", []))
    for rid in ids:
        if rid and str(rid).strip():
            removed.add(str(rid).strip())
    sample["removed_ids"] = sorted(removed)

    selected = sample.get("selected", [])
    if isinstance(selected, list):
        ban = set(str(i).strip() for i in ids if i and str(i).strip())
        for i, obj in enumerate(selected):
            if isinstance(obj, dict) and obj.get("id") in ban:
                selected[i] = None
        sample["selected"] = selected

    write_json(sample_path, sample)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Local title matching (search for phrases) ------------------------------------------------------------------------#####
import re

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def title_matches_phrase(title: str, phrase: str) -> bool:
    """
    True if phrase occurs in title as a contiguous substring (case-insensitive),
    with whitespace normalized.
    """
    if not title or not phrase:
        return False
    t = normalize_text(title)
    p = normalize_text(phrase)
    return p in t

def title_matches_token(title: str, token: str) -> bool:
    """
    True if token occurs anywhere in title (case-insensitive substring).
    No word boundaries. So e.g. "gptase" matches token "gpt".
    """
    if not title or not token:
        return False
    return token.strip().lower() in title.strip().lower()

###--- Candidate retrieval + union + filter ---------------------------------------------------------------------------------###
def union_items_by_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate by article_identity.
    """
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        pid = it.get("article_identity")
        if isinstance(pid, str) and pid.strip():
            if pid in seen:
                continue
            seen.add(pid)
        out.append(it)
    return out

def fetch_candidates_for_tokens(
    *,
    base_params: Dict[str, str],
    tokens: List[str],
    page_limit: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Fetch candidates by querying each token separately via the API title param.
    Then union by article_identity.
    """
    merged: List[Dict[str, Any]] = []
    for tok in tokens:
        tok = str(tok).strip()
        if not tok:
            continue
        p = dict(base_params)
        p["title"] = tok
        res = fetch_all(params=p, page_limit=page_limit, max_pages=max_pages)
        merged.extend(res["items"])
    return union_items_by_id(merged)

def fetch_and_filter_titles(
    *,
    base_params: Dict[str, str],
    must_contain_phrases: List[str],
    must_contain_tokens: List[str],
    page_limit: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:

    candidate_tokens = build_candidate_tokens(must_contain_phrases, must_contain_tokens)

    candidates = fetch_candidates_for_tokens(
        base_params=base_params,
        tokens=candidate_tokens,
        page_limit=page_limit,
        max_pages=max_pages,
    )

    filtered: List[Dict[str, Any]] = []
    for it in candidates:
        title = it.get("title") or ""
        ok = False

        for ph in must_contain_phrases:
            if title_matches_phrase(title, ph):
                ok = True
                break

        if not ok:
            for tok in must_contain_tokens:
                if title_matches_token(title, tok):
                    ok = True
                    break

        if ok:
            filtered.append(it)

    return union_items_by_id(filtered)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Search terms -----------------------------------------------------------------------------------------------------#####
# Title-based matching only. Substring matching is allowed (case-insensitive).
# Phrases are matched as contiguous normalized substrings (see title_matches_phrase).
# Tokens are matched as case-insensitive substrings (see title_matches_token).

SEARCH_PHRASES = [
    # Tier 1: modern LLM phrases
    "language model",
    "foundation model",
    "fine-tuning",
    "generative"
]

SEARCH_TOKENS = [
    # Tier 1: short tokens / common
    "transformer",
    "pretrained",
    "alignment",

    # Tier 2: model family names
    "gpt",
    "bert",

    # Tier 3: method-y correlates
    "self-supervised",
    "transfer learning",
    "representation learning",
    #"text generation",
    "few-shot",
    #"chain-of-thought",
    "agent",

    # Tier 4: NLP tasks / domain phrases (can be noisy)
    #"natural language processing",
    #"named entity recognition",
    "question answering"
]
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Debug: automatic match counting ----------------------------------------------------------------------------------#####
def count_matches(items: List[Dict[str, Any]], phrases: List[str], tokens: List[str]) -> Dict[str, int]:
    """
    Count how many items match each phrase/token in the title.
    """
    counts: Dict[str, int] = {}

    for key in phrases:
        counts[key] = 0
    for key in tokens:
        counts[key] = 0

    for it in items:
        t = it.get("title") or ""
        for ph in phrases:
            if title_matches_phrase(t, ph):
                counts[ph] += 1
        for tok in tokens:
            if title_matches_token(t, tok):
                counts[tok] += 1

    return counts
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Step 1: Create pool file (API) -----------------------------------------------------------------------------------#####
def create_pool_file(
    *,
    # --- Search params (ALL optional) ---
    abstract: Optional[str] = None,
    title: Optional[str] = None,
    unified: Optional[str] = None,
    author: Optional[str] = None,
    journal: Optional[str] = None,
    journal_title: Optional[str] = None,
    keyword: Optional[str] = None,
    articleType: Optional[str] = None,
    subjectArea: Optional[str] = None,
    postedAfter: Optional[str] = None,   # YYYY-MM-DD
    postedBefore: Optional[str] = None,  # YYYY-MM-DD
    status: Optional[str] = None,

    # --- Fetch controls ---
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: Optional[int] = None,

    # --- Output ---
    pool_path: Path = DEFAULT_POOL_PATH,
) -> Dict[str, Any]:
    """
    Create/update pool file from the API.
    Writes pool_path and returns a summary dict.
    """
    params = compact_params({
        "abstract": abstract,
        "title": title,
        "unified": unified,
        "author": author,
        "journal": journal,
        "journal_title": journal_title,
        "keyword": keyword,
        "articleType": articleType,
        "subjectArea": subjectArea,
        "postedAfter": postedAfter,
        "postedBefore": postedBefore,
        "status": status,
    })

    search_terms = {
        "phrases": SEARCH_PHRASES,
        "tokens": SEARCH_TOKENS,
        "candidate_tokens": build_candidate_tokens(SEARCH_PHRASES, SEARCH_TOKENS),
    }

    items = fetch_and_filter_titles(
        base_params=params,
        must_contain_phrases=SEARCH_PHRASES,
        must_contain_tokens=SEARCH_TOKENS,
        page_limit=page_limit,
        max_pages=max_pages,
    )

    counts = count_matches(items, phrases=SEARCH_PHRASES, tokens=SEARCH_TOKENS)
    print("[DEBUG] match_counts:", counts)
    print("[DEBUG] pool_size:", len(items))

    pool_map = build_pool_map(items)

    print(f"[DEBUG] fetched_items={len(items)}")
    print(f"[DEBUG] pool_map_ids={len(pool_map)}")

    write_json(
        pool_path,
        {
            "query_params": params,
            "search_terms": search_terms,
            "count": len(pool_map),
            "pool": pool_map,
        },
    )

    return {
        "query_params": params,
        "search_terms": search_terms,
        "pool_count": len(pool_map),
        "match_counts": counts,
        "pool_path": str(pool_path),
    }
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Step 3: Create sample file ONLY from pool file -------------------------------------------------------------------#####
def create_sample_from_pool_file(
    *,
    n: int,
    pool_path: Path = DEFAULT_POOL_PATH,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
) -> Dict[str, Any]:
    """
    Create/update sample.json based ONLY on the pool.json file.
    No API calls.
    """
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    pool_obj = read_json(pool_path)

    pool_map = pool_obj.get("pool")
    if not isinstance(pool_map, dict) or len(pool_map) == 0:
        raise RuntimeError("Pool file invalid: missing/empty 'pool' map.")

    if sample_path.exists():
        sample_obj = read_json(sample_path)
    else:
        sample_obj = {"n": n, "selected": [], "removed_ids": []}

    sample_obj["n"] = n

    final_sample = refill_sample(sample_obj=sample_obj, pool_map=pool_map, n=n)
    final_sample["query_params"] = pool_obj.get("query_params", {})
    final_sample["search_terms"] = pool_obj.get("search_terms", {})

    write_json(sample_path, final_sample)

    return {
        "pool_count": len(pool_map),
        "sample": final_sample,
        "pool_path": str(pool_path),
        "sample_path": str(sample_path),
    }
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Step 4: Index entry from sample file -------------------------------------------------------------------#####
DEFAULT_INDEX_PATH = Path("data", "papersets_ids", "natureCom", "index.json")


def create_index_entry_from_sample(
    *,
    set_name: str,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    require_all_nature_ids: bool = True,
) -> Dict[str, Any]:
    """
    Step 3) Upsert one set in index.json (Option 1 schema):
    {
      "sets": [
        {
          "name": "...",
          "total": N,
          "paper_ids": [...],
          "paper_meta": {
            "<nature_id>": {"url_pdf":"", "url_peerreview":"", "title":""},
            ...
          }
        }
      ]
    }

    - paper_ids come from sample.selected[*].nature_id
    - paper_meta entries are created with empty strings if missing
    - existing paper_meta values are preserved on reruns
    """
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    sample = read_json(sample_path)

    n = sample.get("n")
    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(f"Sample file invalid: expected positive int 'n', got {n!r}")

    selected = sample.get("selected", [])
    if not isinstance(selected, list):
        raise RuntimeError("Sample file invalid: expected 'selected' to be a list.")

    # --- collect nature_ids in sample order ---
    nature_ids: List[str] = []
    missing: List[str] = []

    for obj in selected:
        if obj is None:
            missing.append("<null>")
            continue
        if not isinstance(obj, dict):
            missing.append("<non-dict>")
            continue

        nat = obj.get("nature_id")
        if isinstance(nat, str) and nat.strip():
            nature_ids.append(nat.strip())
        else:
            rid = obj.get("rs_id") or "<unknown>"
            missing.append(str(rid))

    # de-dupe while preserving order (just in case)
    seen: Set[str] = set()
    deduped: List[str] = []
    for x in nature_ids:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    nature_ids = deduped

    if require_all_nature_ids and missing:
        raise RuntimeError(
            "Some selected items have no nature_id (fill them first). "
            f"Missing for: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    # --- load or init index.json ---
    if index_path.exists():
        index_obj = read_json(index_path)
    else:
        index_obj = {"sets": []}

    if not isinstance(index_obj, dict):
        index_obj = {"sets": []}
    if "sets" not in index_obj or not isinstance(index_obj["sets"], list):
        index_obj["sets"] = []

    # --- find existing set (by name) ---
    existing_set: Optional[Dict[str, Any]] = None
    existing_idx: Optional[int] = None
    for i, s in enumerate(index_obj["sets"]):
        if isinstance(s, dict) and str(s.get("name") or "").strip() == set_name:
            existing_set = s
            existing_idx = i
            break

    existing_meta: Dict[str, Any] = {}
    if isinstance(existing_set, dict):
        pm = existing_set.get("paper_meta")
        if isinstance(pm, dict):
            existing_meta = pm

    # --- build/merge paper_meta, preserving existing filled values ---
    paper_meta: Dict[str, Dict[str, str]] = {}

    for nid in nature_ids:
        prev = existing_meta.get(nid) if isinstance(existing_meta, dict) else None

        url_pdf = ""
        url_peerreview = ""
        title = ""

        if isinstance(prev, dict):
            # keep anything already filled
            if isinstance(prev.get("url_pdf"), str):
                url_pdf = prev.get("url_pdf") or ""
            if isinstance(prev.get("url_peerreview"), str):
                url_peerreview = prev.get("url_peerreview") or ""
            if isinstance(prev.get("title"), str):
                title = prev.get("title") or ""

        paper_meta[nid] = {
            "url_pdf": url_pdf,
            "url_peerreview": url_peerreview,
            "title": title,
        }

    new_entry = {
        "name": set_name,
        "total": n,
        "paper_ids": nature_ids,
        "paper_meta": paper_meta,
    }

    replaced = False
    if existing_idx is not None:
        index_obj["sets"][existing_idx] = new_entry
        replaced = True
    else:
        index_obj["sets"].append(new_entry)

    write_json(index_path, index_obj)

    return {
        "set_name": set_name,
        "total": n,
        "paper_ids_count": len(nature_ids),
        "missing_nature_ids": len(missing),
        "index_path": str(index_path),
        "replaced": replaced,
    }
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Convenience wrapper (API + sample) -------------------------------------------------------------------------------#####
def main(
    *,
    # --- Search params (ALL optional) ---
    abstract: Optional[str] = None,
    title: Optional[str] = None,
    unified: Optional[str] = None,
    author: Optional[str] = None,
    journal: Optional[str] = None,
    journal_title: Optional[str] = None,
    keyword: Optional[str] = None,
    articleType: Optional[str] = None,
    subjectArea: Optional[str] = None,
    postedAfter: Optional[str] = None,   # YYYY-MM-DD
    postedBefore: Optional[str] = None,  # YYYY-MM-DD
    status: Optional[str] = None,

    # --- Sampling ---
    n: int = 50,

    # --- Fetch controls ---
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: Optional[int] = None,

    # --- Outputs ---
    pool_path: Path = DEFAULT_POOL_PATH,
    sample_path: Path = DEFAULT_SAMPLE_PATH,

    # --- Behavior ---
    rebuild_pool: bool = True,
) -> Dict[str, Any]:
    """
    One-call workflow:
    - optionally fetch pool -> save pool.json
    - create/update sample.json from pool.json
    - return summary dict
    """
    if rebuild_pool or not pool_path.exists():
        create_pool_file(
            abstract=abstract,
            title=title,
            unified=unified,
            author=author,
            journal=journal,
            journal_title=journal_title,
            keyword=keyword,
            articleType=articleType,
            subjectArea=subjectArea,
            postedAfter=postedAfter,
            postedBefore=postedBefore,
            status=status,
            page_limit=page_limit,
            max_pages=max_pages,
            pool_path=pool_path,
        )

    return create_sample_from_pool_file(
        n=n,
        pool_path=pool_path,
        sample_path=sample_path,
    )
#####------------------------------------------------------------------------------------------------------------------------#####



#To read: "rs-4959578", "rs-5327690", "rs-4582961",
#Int: "rs-6672054",
bannedIDs_topic = ["rs-5045672", "rs-3960210", "rs-3951309", "rs-5061922", "rs-4218656", "rs-4959578", "rs-3749314",
                   "rs-4547474", "rs-4463391", "rs-3979948", "rs-5039845", "rs-4800799", "rs-5683637", "rs-5084780",
                   "rs-6681474", "rs-3811693", "rs-3796284", "rs-5537857", "rs-6908218", "rs-5158614", "rs-4831323",
                   "rs-6193835", "rs-4984522", "rs-5912049", "rs-4045626", "rs-5776626", "rs-4203122", "rs-4718479",
                   "rs-2884955", "rs-5353095", "rs-5530535", "rs-4715536", "rs-6598384", "rs-5374449", "rs-5456413",
                   "rs-6237413", "rs-5973632", "rs-6272325", "rs-3964901", "rs-4209550", "rs-6033997", "rs-5459798",
                   "rs-3957695", "rs-3981393", "rs-5904229", "rs-3878828", "rs-6310016", "rs-5187157", "rs-4865151",
                   "rs-5235151", "rs-4780630", "rs-4419425", "rs-3443805", "rs-5811311", "rs-3873286", "rs-4466481",
                   "rs-5327690", "rs-4133186", "rs-3788577", "rs-5522083", "rs-5522707", "rs-6702895", "rs-3898201",
                   "rs-4905116", "rs-3909102", "rs-4393432", "rs-5391023", "rs-4767509", "rs-4250821", "rs-5367202",
                   "rs-5850681", "rs-4157720", "rs-5698716", "rs-4158621", "rs-4277811", "rs-5397645", "rs-4672609",
                   "rs-3868268", "rs-4326648", "rs-5050970", "rs-5059066", "rs-4908650", "rs-5001134", "rs-6672054",
                   "rs-5473963", "rs-4380373", "rs-4998867", "rs-6127712", "rs-3979270", "rs-6406486", "rs-5595029",
                   "rs-3999734", "rs-4582961", "rs-5633068", "rs-3808475", "rs-4656898", "rs-3893234", "rs-5194931",
                   "rs-3876111", "rs-6074849", "rs-6521849", "rs-4704369", "rs-4682574", "rs-3954760", "rs-4308500",
                   "rs-5922891"
                   ]
#maybe ban: "rs-3671637", "rs-4509973", "rs-5835917",
bannedIDs_noReview = []

# ban_ids(ids=bannedIDs_topic + bannedIDs_noReview)

# --- Run example: split workflow ---
# 1) Pool from API
"""
create_pool_file(
    journal_title="Nature Communications",
    postedAfter="2024-01-01",
    postedBefore="2026-01-01",
    pool_path=DEFAULT_POOL_PATH,
)
"""

# 2) Manually add the Nature IDs

# 3) Sample from pool (NO API)
out = create_sample_from_pool_file(
    n=samplesize,
    pool_path=DEFAULT_POOL_PATH,
    sample_path=DEFAULT_SAMPLE_PATH,
)

print([(x["rs_id"], x["title"], x["link"]) for x in out["sample"]["selected"] if x is not None])

idx_summary = create_index_entry_from_sample(
    set_name="NatureCom_" + str(setname) + "_" + str(samplesize),
    sample_path=DEFAULT_SAMPLE_PATH,
    index_path=DEFAULT_INDEX_PATH,
    require_all_nature_ids=True,
)

print("[DEBUG] index_update:", idx_summary)