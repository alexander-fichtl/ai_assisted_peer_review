"""
Builds the medical journal list for the survey on AI-assisted peer review policies.

Source: Clarivate JCR 2024 (jcr.clarivate.com). Selection: top 8 journals by 2024
JIF from 6 Clinical Medicine subcategories, deduplicated by ISSN/eISSN. Uniform
quota (8 per subcategory) removes any per-subcategory weighting.

Input: CSV exports in medical_jcr_exports/ - one file per subcategory, exported from
JCR via Journals -> filter to subcategory -> Export CSV.
Output: medical_journals.csv
"""

import csv
from pathlib import Path


EXPORT_DIR = Path(__file__).parent / "medical_jcr_exports"
OUTPUT_CSV = Path(__file__).parent / "medical_journals.csv"

N_PER_CAT = 8  # uniform top-N per subcategory; 6 subcategories -> 48 total

SUBCATEGORIES = [
    "MEDICINE, GENERAL & INTERNAL",
    "ONCOLOGY",
    "CARDIAC & CARDIOVASCULAR SYSTEMS",
    "CLINICAL NEUROLOGY",
    "RADIOLOGY, NUCLEAR MEDICINE & MEDICAL IMAGING",
    "HEALTH CARE SCIENCES & SERVICES",
]

FIELDNAMES = [
    "name",
    "acronym",            # JCR Abbreviation
    "jif_2024",
    "total_citations",
    "category",           # which Clinical Medicine subcategory the journal was picked from
    "field",              # always "Healthcare"
    "issn",
    "eissn",
    "publisher",
    "website_url",
    "url_status",
    "policy_category",    # AI review prohibited | partially_allowed | actively_used | no_policy
    "policy_source",
    "policy_quote",
    "notes",
    "checked",            # yes | partial | empty (Chrome-verification audit trail)
]


def parse_jcr_csv(path):
    """
    Parse a JCR export. The CSV format is non-standard:
      line 1: "Journal Data Filtered By: Selected Categories: ..."
      line 2: blank
      line 3: column headers
      lines 4..N: data rows
      last 2 lines: copyright/footer
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("Journal name,")), None)
    if header_idx is None:
        return []

    journals = []
    reader = csv.reader(lines[header_idx:])
    header = next(reader)
    for row in reader:
        if not row or not row[0]:
            continue
        if row[0].startswith("Copyright") or row[0].startswith("By exporting"):
            continue
        if len(row) < 12:
            continue
        d = dict(zip(header, row))

        #numeric coercions for sorting
        try:
            d["total_citations_int"] = int(d["Total Citations"].replace(",", ""))
        except (ValueError, KeyError):
            d["total_citations_int"] = 0

        jif = d.get("2024 JIF", "")
        if jif in ("N/A", ""):
            d["jif_float"] = None
        elif jif.startswith("<"):
            d["jif_float"] = 0.05  # "<0.1" -> treat as a small positive
        else:
            try:
                d["jif_float"] = float(jif)
            except ValueError:
                d["jif_float"] = None

        journals.append(d)
    return journals


def real_ids(j):
    """ISSN/eISSN values, skipping placeholder 'N/A' entries."""
    ids = set()
    for k in ("ISSN", "eISSN"):
        v = j.get(k, "").strip()
        if v and v != "N/A":
            ids.add(v)
    return ids


def main():
    # load all per-subcategory exports, indexed by category name
    by_category = {}
    for path in sorted(EXPORT_DIR.glob("*.csv")):
        journals = parse_jcr_csv(path)
        if journals:
            cat = journals[0].get("Category", "?")
            by_category[cat] = journals
            print(f"  loaded {cat}: {len(journals)} journals  ({path.name})")

    missing = set(SUBCATEGORIES) - set(by_category)
    if missing:
        print(f"\nWARNING: missing exports for: {sorted(missing)}")

    # pick top-N by JIF per subcategory, dedupe by ISSN/eISSN
    seen_ids = set()
    selected = []
    for cat in SUBCATEGORIES:
        candidates = sorted(
            by_category.get(cat, []),
            key=lambda j: (j["jif_float"] or 0),
            reverse=True,
        )
        picked = 0
        for j in candidates:
            if picked >= N_PER_CAT:
                break
            ids = real_ids(j)
            if ids & seen_ids:
                continue
            seen_ids |= ids
            j["_picked_category"] = cat
            selected.append(j)
            picked += 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for j in selected:
            eissn = j.get("eISSN", "")
            writer.writerow({
                "name":             j["Journal name"],
                "acronym":          j["JCR Abbreviation"],
                "jif_2024":         j["2024 JIF"],
                "total_citations":  j["Total Citations"],
                "category":         j["_picked_category"],
                "field":            "Healthcare",
                "issn":             j.get("ISSN", ""),
                "eissn":            eissn if eissn != "N/A" else "",
                "publisher":        j.get("Publisher", ""),
                "website_url":      "",
                "url_status":       "",
                "policy_category":  "",
                "policy_source":    "",
                "policy_quote":     "",
                "notes":            "",
                "checked":          "",
            })

    print(f"\nWrote {len(selected)} journals -> {OUTPUT_CSV}")
    print(f"fill-rate per subcategory (target = {N_PER_CAT}):")
    by_picked = {}
    for j in selected:
        by_picked.setdefault(j["_picked_category"], 0)
        by_picked[j["_picked_category"]] += 1
    for cat in SUBCATEGORIES:
        actual = by_picked.get(cat, 0)
        flag = "" if actual == N_PER_CAT else "  <- short"
        print(f"  {cat}  {actual}/{N_PER_CAT}{flag}")


if __name__ == "__main__":
    main()
