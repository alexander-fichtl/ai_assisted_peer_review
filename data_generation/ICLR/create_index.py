import json
from pathlib import Path

name = "ICLR26_rndm50" 
conference = "ICLR.cc/2026/Conference"
paper_ids = ["gL2plIZ4Sf", "CWpQsAubxy", "owFVvaLFdw", "IGcGUHEkM5", "mpAMH1OyMO", "7y11BdJIOp", "rv0j1CyRQK", "KMFotFeGjJ", "gxkOfdaaMA", "ooYtHcj6LI", "w0rVXs6QJM", "EFOMez9cBW", "kVojSLUcvS", "liDUSnwUVP", "Obefq4k8iG", "sDlIWuHaRt", "WNHNfkulYA", "8uZ5UdIul2", "koIbbsfKSf", "gpeXJ9UI07", "xmsbFPmsOz", "6K08FPo2cf", "WrEQFwWCdT", "084SvT55yk", "ugGIpVfGE6", "Ba0RlUn0TR", "lAlI5FuIf7", "BZ1SSMS7L6", "KfjpyOcPQj", "aQia5HdgBY", "jKcHPxRvk9", "Ovmvu3IZAp", "jfhIbJ3K8e", "ifC1qxP5rS", "dOxOX4GF1k", "OWHKdYwYiF", "IljH2Bt6jO", "Hgw56DUFzD", "CYmjrbQRyM", "zZNYUkBS77", "GP5RHZnEsw", "KJ0ktHAuds", "rPrNaDJrLx", "PS1YS8Wv4t", "1RrwTXSU4c", "fdpB9RBFv5", "5IqvMNQaUg", "XCW1l9qcxy", "nBta7psl5J", "dc90uPqxWF"]

out_dir = Path("data/final_datasets") / name
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "index.json").write_text(
    json.dumps(
        {"conference": conference, "name": name, "total": len(paper_ids), "paper_ids": paper_ids},
        ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)