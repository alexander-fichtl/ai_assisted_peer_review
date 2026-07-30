"""
Builds the AI/NLP conference list for the survey on AI-assisted peer review policies.

Source: CORE 2023 export (portal.core.edu.au). Filter: rank in {A*, A} AND any
FOR code in {4602, 4611, 4603, 4605}. No manual exclusions - the set is whatever
CORE's taxonomy yields.

URL enrichment via WikiCFP by acronym (future -> current -> all years). MANUAL_URLS
overrides venues where WikiCFP returns stale/wrong results (biennial conferences or
ones not yet posted for the current cycle).

Output: conference_db.csv
Usage:  python build_conference_db.py [--no-urls]
"""

import csv
import io
import re
import sys
import time
import requests
import argparse
from datetime import date


CORE_EXPORT_URL = (
    "https://portal.core.edu.au/conf-ranks/"
    "?search=&by=all&source=CORE2023&sort=arank&page=1&do=Export"
)

WIKICFP_CFP = "http://wikicfp.com/cfp/servlet/event.showcfp?eventid={eid}"

#search order: future editions first, then current year, then all
WIKICFP_SEARCH_YEARS = ["f", "t", "a"]

TARGET_RANKS = {"A*", "A"}
TARGET_FOR_CODES = {
    "4602",  # Artificial Intelligence
    "4611",  # Machine Learning
    "4603",  # Computer Vision, Multimedia & Image Processing
    "4605",  # Data Management & Data Science
}

# manual overrides for venues where WikiCFP returns stale/wrong results
MANUAL_URLS = {
    "AISTATS":     ("https://aistats.org/aistats2026/",         "ok"),
    "CVPR":        ("https://cvpr.thecvf.com/Conferences/2026", "ok"),
    "ICCV":        ("https://iccv.thecvf.com/Conferences/2025", "ok"),  # biennial, odd years
    "Interspeech": ("https://interspeech2026.org/",             "ok"),
    "NAACL":       ("https://2026.naacl.org/",                  "ok"),
    "WWW":         ("https://www2026.thewebconf.org/",          "ok"),
    "ECAI":        ("https://ecai2026.org/",                    "ok"),  # biennial, even years
}

REQUEST_DELAY = 1.5
REQUEST_TIMEOUT = 15

FIELDNAMES = [
    "name",
    "acronym",
    "core_rank",
    "for_codes",
    "field",
    "website_url",
    "url_status",
    "policy_category",
    "policy_source",
    "policy_quote",
    "notes",
]


def download_core(url):
    """Download CORE CSV and keep rows where rank in TARGET_RANKS AND any FOR in TARGET_FOR_CODES."""
    print("Downloading CORE 2023 export...")
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    rows = list(csv.reader(io.StringIO(resp.text)))
    print(f"  {len(rows)} total venues in CORE 2023")

    # CORE CSV columns (no header row):
    # 0=id, 1=title, 2=acronym, 3=source, 4=rank, 5=dblp, 6=for1, 7=for2, 8=for3
    results = {}
    for row in rows:
        if len(row) < 5:
            continue

        rank = row[4].strip()
        if rank not in TARGET_RANKS:
            continue

        for_codes = [row[i].strip() for i in range(6, min(9, len(row))) if row[i].strip()]
        if not any(fc in TARGET_FOR_CODES for fc in for_codes):
            continue

        acronym = row[2].strip()
        if acronym in MANUAL_URLS:
            website_url, url_status = MANUAL_URLS[acronym]
        else:
            website_url, url_status = "", ""

        results[acronym] = {
            "name":            row[1].strip(),
            "acronym":         acronym,
            "core_rank":       rank,
            "for_codes":       ", ".join(for_codes),
            "field":           "AI/NLP",
            "website_url":     website_url,
            "url_status":      url_status,
            "policy_category": "",
            "policy_source":   "",
            "policy_quote":    "",
            "notes":           "",
        }

    n_astar = sum(1 for v in results.values() if v["core_rank"] == "A*")
    n_a = sum(1 for v in results.values() if v["core_rank"] == "A")
    print(f"  {len(results)} venues passed filter  (A*={n_astar}, A={n_a})")
    return results


def extract_event_id(html):
    match = re.search(r'event\.showcfp\?eventid=(\d+)', html)
    return match.group(1) if match else None


def extract_website_url(html):
    match = re.search(r'Link:\s*<a href="([^"]+)"', html)
    return match.group(1).strip() if match else None


def get_wikicfp_url(acronym):
    """Try year=f, year=t, year=a in order; return the first website URL found."""
    for year in WIKICFP_SEARCH_YEARS:
        search_url = f"http://wikicfp.com/cfp/servlet/tool.search?q={acronym}&year={year}"
        try:
            resp = requests.get(search_url, timeout=REQUEST_TIMEOUT,
                                headers={"User-Agent": "Mozilla/5.0 (research scraper)"})
            resp.raise_for_status()
            eid = extract_event_id(resp.text)
            if not eid:
                time.sleep(REQUEST_DELAY)
                continue

            time.sleep(REQUEST_DELAY)
            cfp_resp = requests.get(WIKICFP_CFP.format(eid=eid),
                                    timeout=REQUEST_TIMEOUT,
                                    headers={"User-Agent": "Mozilla/5.0 (research scraper)"})
            cfp_resp.raise_for_status()
            url = extract_website_url(cfp_resp.text)
            if url:
                return url
            time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            print(f"    WikiCFP request failed for {acronym}: {e}")
            time.sleep(REQUEST_DELAY)

    return ""


def year_in_url(url):
    m = re.search(r'20(2[0-9])', url)
    return int(m.group(0)) if m else None


def enrich_with_urls(venues):
    total = len(venues)
    current_year = date.today().year
    for i, (acronym, venue) in enumerate(venues.items(), 1):
        print(f"  [{i}/{total}] {acronym}", end="  ", flush=True)
        if venue["url_status"] == "ok":
            print(f"{venue['website_url']}  [manual]")
            continue
        url = get_wikicfp_url(acronym)
        venue["website_url"] = url
        if not url:
            venue["url_status"] = "not_found"
        else:
            yr = year_in_url(url)
            if yr and yr < current_year - 1:
                venue["url_status"] = "stale"
            else:
                venue["url_status"] = "ok"
        print(f"{url or '(not found)'}  [{venue['url_status']}]")
        time.sleep(REQUEST_DELAY)


def write_csv(venues, path):
    rank_order = {"A*": 0, "A": 1, "B": 2}
    rows = sorted(venues.values(),
                  key=lambda r: (rank_order.get(r["core_rank"], 9), r["acronym"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {path}")


def print_summary(venues):
    rank_order = {"A*": 0, "A": 1}
    rows = sorted(venues.values(),
                  key=lambda r: (rank_order.get(r["core_rank"], 9), r["acronym"]))
    astar = sum(1 for r in rows if r["core_rank"] == "A*")
    a = sum(1 for r in rows if r["core_rank"] == "A")
    with_url = sum(1 for r in rows if r["website_url"])

    print(f"\nConference DB - {date.today()}")
    print(f"filter: rank in {sorted(TARGET_RANKS)}, any FOR in {sorted(TARGET_FOR_CODES)}")
    print(f"total: {len(rows)}  (A*={astar}, A={a}),  with URL: {with_url}\n")
    for r in rows:
        url_short = r["website_url"][:50] if r["website_url"] else "(missing)"
        print(f"{r['core_rank']}\t{r['acronym']}\t{r['for_codes']}\t{url_short}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-urls", action="store_true",
                        help="Skip WikiCFP URL enrichment (faster, offline)")
    args = parser.parse_args()

    venues = download_core(CORE_EXPORT_URL)

    if not args.no_urls:
        print(f"\nFetching website URLs from WikiCFP "
              f"({len(venues)} lookups, ~{len(venues)*REQUEST_DELAY*2:.0f}s)...")
        enrich_with_urls(venues)
    else:
        print("\nSkipping URL enrichment (--no-urls)")

    out_path = "conference_db.csv"
    write_csv(venues, out_path)
    print_summary(venues)


if __name__ == "__main__":
    main()
