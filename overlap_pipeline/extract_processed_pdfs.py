"""
One time extractor for the papers  (step 0)

pulls the paper markdown out of
sc_datasets-main/data/processed_pdfs.zip into overlap_pipeline/data/processed_pdfs/,
skipping the multi-megabyte docling .json files and the _artifacts/ image dumps.

resulting layout:
data/processed_pdfs/ICLR/<paper_id>/<paper_id>.md   (and .txt)
data/processed_pdfs/natureCom/<paper_id>/<paper_id>.md
"""

import argparse
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_ZIP = _HERE.parent / "sc_datasets-main" / "data" / "processed_pdfs.zip"
DEST_ROOT = _HERE / "data" / "processed_pdfs"

#these extensions are pulled out of the archive.
KEEP_SUFFIXES = (".md", ".txt")


def extract(zip_path, dest_root, *, overwrite=False):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    dest_root.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "skipped": 0, "bytes": 0}

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not name.lower().endswith(KEEP_SUFFIXES): #ignore ths one
                continue
            if "_artifacts/" in name: #ignore ths one
                continue

            # "processed_pdfs/ICLR/<id>/<id>.md" -> "ICLR/<id>/<id>.md"
            rel = name.split("processed_pdfs/", 1)[-1]
            out_path = dest_root / rel
            if out_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src:
                data = src.read()
            out_path.write_bytes(data)
            stats["written"] += 1
            stats["bytes"] += len(data)

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                    help=f"Path to processed_pdfs.zip (default: {DEFAULT_ZIP})")
    ap.add_argument("--dest", type=Path, default=DEST_ROOT,
                    help=f"Destination root (default: {DEST_ROOT})")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files instead of skipping them.")
    args = ap.parse_args()

    print(f"Extracting {KEEP_SUFFIXES} from {args.zip}")
    print(f"  -> {args.dest}")
    stats = extract(args.zip, args.dest, overwrite=args.overwrite)
    print(f"  written: {stats['written']}, skipped: {stats['skipped']}, size: {stats['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
