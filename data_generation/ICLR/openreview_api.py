"""
dump_venue() fetches all public submissions for a venue (e.g., ICLR.cc/2025/Conference) from OpenReview v2.
    Stores a summary csv and one file per paper (<id>.json):
        data/
            openreview_basesets/
            <VENUE_ID>/
                submissions.csv
                submissions.jsonl
                submissions_jsons/
                    <id>.json


get(path) loads a <id>.json and get_xxx helpers operate on the loaded object. 
"""



###--- Imports ----------------------------------------------------------------------------------------###
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import openreview
###----------------------------------------------------------------------------------------------------###



###----- Variables ------------------------------------------------------------------------------------###
VAR_venue = "ICLR.cc/2026/Conference" #TODO:CHANGE TO CURRENT CONFERENCE!
OPENREVIEW_API = "https://api2.openreview.net"
OPENREVIEW_WEB = "https://openreview.net"
###----------------------------------------------------------------------------------------------------###



###----- Helper Functions -----------------------------------------------------------------------------###
def unwrap(val, default=""):
    """OpenReview v2 content fields are often {'value': ...} or {'values': [...]}. Return the underlying value."""
    if isinstance(val, dict):
        if "value" in val:
            return val.get("value", default)
        if "values" in val:
            return val.get("values", default)
        return default
    return default if val is None else val

def iter_replies(note: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield each direct reply with a normalized invitation token on _normalized_invitation."""
    for r in ((note.get("details") or {}).get("directReplies") or []):
        inv = r.get("invitation")
        if not inv:
            invs = r.get("invitations") or []
            inv = invs[0] if invs else ""
        rr = dict(r)
        rr["_normalized_invitation"] = str(inv).lower()
        yield rr

def note_to_dict(n) -> Dict[str, Any]:
    """Convert Note object to plain dict, preserving `details` if present."""
    d = n.to_json() if hasattr(n, "to_json") else dict(n)
    if getattr(n, "details", None) and "details" not in d:
        d["details"] = n.details
    return d

def sort_key(n: Dict[str, Any]) -> Tuple[int, str]:
    c = n.get("content") or {}
    raw_num = n.get("number")
    if raw_num is None:
        raw_num = unwrap(c.get("paper_number"), None)
    try:
        numi = int(raw_num) if raw_num is not None else 10**9
    except Exception:
        numi = 10**9
    t = unwrap(c.get("title"), "")
    return (numi, (t.lower() if isinstance(t, str) else str(t).lower()))
###----------------------------------------------------------------------------------------------------###



###----- get() Functions ------------------------------------------------------------------------------###
def get(path: str | Path) -> Dict[str, Any]:
    """Load a submission.json file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_reviews(note: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in iter_replies(note):
        inv = r["_normalized_invitation"]
        if "official_review" in inv or inv.endswith("/-/official_review") or "review" in inv:
            out.append(r)
    return out

def get_metareviews(note: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in iter_replies(note):
        inv = r["_normalized_invitation"]
        if "meta_review" in inv or "metareview" in inv or "area_chair_review" in inv:
            out.append(r)
    return out

def get_decisions(note: Dict[str, Any]) -> List[Dict[str, Any]]: 
    out: List[Dict[str, Any]] = [] 
    for r in iter_replies(note): 
        if "decision" in r["_normalized_invitation"]: 
            out.append(r) 
    return out

def get_bucket(note: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for d in get_decisions(note):
        for v in (d.get("content") or {}).values():
            pieces.append(str(unwrap(v, "")))
    blob = " ".join(pieces).lower()
    if "withdraw" in blob:
        return "withdrawn_submissions"
    if "desk" in blob and "reject" in blob:
        return "desk_rejected_submissions"
    if "accept" in blob:
        if "oral" in blob:
            return "accepted (oral)"
        if "spotlight" in blob:
            return "accepted (spotlight)"
        if "poster" in blob:
            return "accepted (poster)"
        return "accepted (poster)"
    if "reject" in blob:
        return "rejected"
    return "unknown"

def get_title(note: Dict[str, Any]) -> str:
    return unwrap((note.get("content") or {}).get("title"), "")

def get_links(note: Dict[str, Any]) -> Dict[str, str]:
    forum = note.get("forum", note.get("id"))
    pdf_rel = unwrap((note.get("content") or {}).get("pdf"), "")
    supp_rel = unwrap((note.get("content") or {}).get("supplementary_material"), "")
    def abs_url(rel: str) -> str:
        return f"{OPENREVIEW_WEB}{rel}" if isinstance(rel, str) and rel.startswith("/") else rel
    return {
        "forum_url": f"{OPENREVIEW_WEB}/forum?id={forum}",
        "pdf_url": abs_url(pdf_rel),
        "supplementary_url": abs_url(supp_rel) if supp_rel else "",
    }
###----------------------------------------------------------------------------------------------------###



###----- Functions ------------------------------------------------------------------------------------###
def dump_venue(venue_id: str, out_root: Path):
    print(f"[info] Connecting to OpenReview API: {OPENREVIEW_API}")
    #client = openreview.api.OpenReviewClient(baseurl=OPENREVIEW_API)

    print(f"[info] Fetching public notes for venueid={venue_id} (with directReplies)…")
    #notes = client.get_all_notes(content={'venueid': venue_id}, details='directReplies')
    invitations = [
        f"{venue_id}/-/Blind_Submission",   
        f"{venue_id}/-/Submission",
        #Include non-accepted ones:
        f"{venue_id}/-/Withdrawn_Submission",
        f"{venue_id}/-/Desk_Rejected",
        f"{venue_id}/-/Rejected_Submission",
    ]

    notes = []
    seen = set()
    for inv in invitations:
        batch = openreview.api.OpenReviewClient(baseurl=OPENREVIEW_API)\
                .get_all_notes(invitation=inv, details='directReplies')
        for n in batch or []:
            nid = getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
            if nid and nid not in seen:
                seen.add(nid)
                notes.append(n)
    notes = [note_to_dict(n) for n in notes]
    notes.sort(key=sort_key)
    print(f"[info] Retrieved {len(notes)} notes")

    outputfolder = out_root / venue_id.replace('/', '_')
    outputfolder.mkdir(parents=True, exist_ok=True)

    csv_path = outputfolder / "submissions.csv"
    jsonl_path = outputfolder / "submissions.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv, \
         jsonl_path.open("w", encoding="utf-8") as fjl:

        writer = csv.DictWriter(
            fcsv,
            fieldnames=["id", "title", "venue", "num_reviews", "num_metareviews",
                        "status_bucket", "forum_url", "pdf_url"
            ],
        )
        writer.writeheader()

        for note in notes:
            content = note.get("content") or {}
            links = get_links(note)

            row = {
                "id": note.get("id"),
                "title": get_title(note),
                "venue": unwrap(content.get("venue"), None) or unwrap(content.get("venueid"), None) or venue_id,
                "num_reviews": len(get_reviews(note)),
                "num_metareviews": len(get_metareviews(note)),
                "status_bucket": get_bucket(note),
                "forum_url": links["forum_url"],
                "pdf_url": links["pdf_url"],
            }

            writer.writerow(row)
            fjl.write(json.dumps(row, ensure_ascii=False) + "\n")

            pdir = outputfolder / "submissions_jsons"
            pdir.mkdir(parents=True, exist_ok=True)
            with (pdir / (str(note.get("id")) + ".json")).open("w", encoding="utf-8") as f:
                json.dump(note, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote:")
    print(f"  - {csv_path}")
    print(f"  - {jsonl_path}")
    print(f"  - {outputfolder}/<paper_number_or_forum>/submission.json")
    print(f"Total papers: {len(notes)}")

def list_paper_ids(outputfolder: Path) -> List[str]:
    """
    Return all paper IDs based on files in:
      <outputfolder>/submissions_jsons/<id>.json
    """
    pdir = outputfolder / "submissions_jsons"
    if not pdir.exists():
        return []
    ids = [p.stem for p in pdir.glob("*.json")]
    ids.sort(key=lambda x: (len(x), x))
    return ids
###----------------------------------------------------------------------------------------------------###



###----- Main -----------------------------------------------------------------------------------------###
def main(venue):
    out_root = Path("data/openreview_basesets") 
    out_root.mkdir(parents=True, exist_ok=True)
    outputfolder = out_root / venue.replace('/', '_')
    dump_venue(venue_id=venue, out_root=out_root)

    ids = list_paper_ids(outputfolder)
    #print(f"\nPaper IDs in {outputfolder}:\n{ids}")
    print(f"\nAmount of Paper IDs in {outputfolder}:\n{len(ids)}")

main(venue=VAR_venue)
print("DONE :)")
###----------------------------------------------------------------------------------------------------###