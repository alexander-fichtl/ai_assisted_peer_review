import json
from pathlib import Path

def patch_formulas(pid: str, root: str = "data/processed_pdfs/natureCom"):
    base = Path(root) / pid
    json_p = base / f"{pid}.json"
    md_p = base / f"{pid}.md"
    txt_p = base / f"{pid}.txt"

    data = json.loads(json_p.read_text(encoding="utf-8"))

    formulas = []
    for item in data.get("texts", []):
        label = (item.get("label") or "").strip()
        orig = (item.get("orig") or "").strip()
        if label == "formula" and orig:
            formulas.append(orig)

    marker = "<!-- formula-not-decoded -->"

    for file_p in [md_p, txt_p]:
        text = file_p.read_text(encoding="utf-8")
        count = text.count(marker)
        if count != len(formulas):
            print(f"[WARN] {pid} in {file_p.name}: {count} markers but {len(formulas)} formulas — skipping")
            continue
        for formula in formulas:
            text = text.replace(marker, formula, 1)
        file_p.write_text(text, encoding="utf-8")

    print(f"[ok] {pid}: patched {len(formulas)} formulas")

"""
paper_ids = [
    "gL2plIZ4Sf",
    "CWpQsAubxy",
    "owFVvaLFdw",
    "IGcGUHEkM5",
    "mpAMH1OyMO",
    "7y11BdJIOp",
    "rv0j1CyRQK",
    "KMFotFeGjJ",
    "gxkOfdaaMA",
    "w0rVXs6QJM",
    "EFOMez9cBW",
    "kVojSLUcvS",
    "liDUSnwUVP",
    "Obefq4k8iG",
    "sDlIWuHaRt",
    "WNHNfkulYA",
    "8uZ5UdIul2",
    "koIbbsfKSf",
    "gpeXJ9UI07",
    "xmsbFPmsOz",
    "6K08FPo2cf",
    "WrEQFwWCdT",
    "084SvT55yk",
    "ugGIpVfGE6",
    "Ba0RlUn0TR",
    "lAlI5FuIf7",
    "BZ1SSMS7L6",
    "KfjpyOcPQj",
    "aQia5HdgBY",
    "jKcHPxRvk9",
    "Ovmvu3IZAp",
    "ifC1qxP5rS",
    "dOxOX4GF1k",
    "OWHKdYwYiF",
    "IljH2Bt6jO",
    "Hgw56DUFzD",
    "CYmjrbQRyM",
    "zZNYUkBS77",
    "GP5RHZnEsw",
    "KJ0ktHAuds",
    "rPrNaDJrLx",
    "PS1YS8Wv4t",
    "1RrwTXSU4c",
    "fdpB9RBFv5",
    "5IqvMNQaUg",
    "XCW1l9qcxy",
    "nBta7psl5J",
    "dc90uPqxWF"
  ]
  """
paper_ids = [
    "s41467-025-61040-5",
    "s41467-025-64681-8",
    "s41467-025-67178-6",
    "s41467-025-61599-z",
    "s41467-025-58872-6",
    "s41467-025-61684-3",
    "s41467-025-64574-w",
    "s41467-025-66644-5",
    "s41467-024-55146-5",
    "s41467-024-53742-z",
    "s41467-025-63846-9",
    "s41467-025-64894-x",
    "s41467-025-57473-7",
    "s41467-024-53457-1",
    "s41467-024-49798-6",
    "s41467-025-66049-4",
    "s41467-024-51669-z",
    "s41467-024-53287-1",
    "s41467-025-64227-y",
    "s41467-025-66620-z",
    "s41467-025-66507-z",
    "s41467-025-56625-z",
    "s41467-025-57426-0",
    "s41467-025-67791-5",
    "s41467-025-61309-9",
    "s41467-025-55920-z",
    "s41467-024-53768-3",
    "s41467-025-63160-4",
    "s41467-025-60989-7",
    "s41467-025-64105-7",
    "s41467-025-61754-6"
]

for pid in paper_ids:
    patch_formulas(pid)