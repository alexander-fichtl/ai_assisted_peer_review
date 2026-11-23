## Data

Structure of the data folder is as follows:

```
data/
├── context/                      # Static reference text fed to LLM reviewers: Reviewer guide, code of conduct, statistics, etc.
├── final_datasets/               # Curated experiment datasets 
│   ├── ICLR26_rndm50/
│   │   ├── dataset.csv           # 50 sampled ICLR 2026 papers + metadata + human reviews
│   │   └── index.json
│   └── natureCom_nlpterms_31/
│       └── dataset.csv           # 31 Nature Communications NLP papers + metadata
├── papersets_ids/                # Paper selection indices — natureCom only 
│   └── natureCom/
│       ├── pool_nlpterms.json    # Full pool from Research Square API
│       ├── sample_nlpterms_31.json
│       └── index.json            # Maps paper IDs to PDF/review links (includes manual entries)
├── reviews_human.zip             # Scraped human reviews — natureCom only; ICLR reviews can be generated with the respective script in the review_generation/ folder
├── llmjudge_evals.zip            # LLM-as-a-judge outputs: CSV and per-paper judgment JSONs for each review source
├── processed_pdfs/               # Docling output 
│   ├── ICLR/
│   └── natureCom/
│       └── <paper_id>/
│           ├── <paper_id>.json
│           ├── <paper_id>.md
│           ├── <paper_id>.txt
│           └── <paper_id>_artifacts/   # Extracted images (PNG)
└── reviews_generated/            # LLM-generated reviews 
```
Some large files containing the PDFs and images of all papers had to be temporarily excluded due to file size constraints. If they haven't been uploaded yet, you can message us at alexander.fichtl@tum.de, and we will provide them. 

### Different layouts of ICLR and natureCom explained:

- **ICLR**: Human reviews come from OpenReview and are fetched together with all other paper metadata. They live directly inside `final_datasets/ICLR26_rndm50/dataset.csv`. There is no separate `reviews_human/ICLR/` folder. `papersets_ids/ICLR/` also does not exist — paper selection is handled inside `dataset_generator_ICLR.py`.

- **natureCom**: While the paper PDFs are from Research Square, the human reviews had to be scraped separately from Nature Communications and are stored as individual JSONs in `reviews_human/natureCom/`. Paper selection IDs and PDF/review links are tracked separately in `papersets_ids/natureCom/`.

### reviews_generated file structure

Each paper folder under `gpt/`, `llama/`, or `qwen/` contains:
```
<paper_id>/
├── final_review.json             # Review per question (HTML-renderable)
├── final_review_openreview.json  # OpenReview-style content block
├── final_review.html             # Human-readable HTML
├── final_review.txt              # Plain text
├── audit.log                     # Token counts per question
└── per_question/
    └── q01.json ... q11.json     # Full prompt + answer per question (ICLR only)
```

### reviews_human — review format

`reviews_<paper_id>.json` (natureCom only) contains the first round of peer reviews:
```json
{
    "first_remarks": [
        {"reviewer_id": "...", "remarks_to_the_author": "..."},
        {"reviewer_id": "...", "remarks_to_the_author": "...", "invalid": "yes"},
        ...
    ]
}
```
Reviews marked `"invalid": "yes"` are duplicates or co-authored reviews — exclude them when loading.

---

### Data Ownership, Licensing, and Attribution

This repository contains publicly available scientific submissions and peer review reports collected for academic research purposes. The original manuscripts and human-written reviews were not created by the authors of this repository, and we do not claim ownership or copyright over these materials.

**ICLR 2026:** The included submissions and reviews were obtained from [OpenReview](https://openreview.net/). We verified that all included submission records explicitly specify a Creative Commons Attribution 4.0 International (CC BY 4.0) license and are publicly accessible. The associated reviews are also distributed under CC BY 4.0.

**Nature Communications:** The included peer review reports were obtained from publicly available review materials published by Nature Communications.

**Research Square:** Any included preprint materials remain subject to their original licenses, which should be consulted through the corresponding source records.

All third-party materials remain subject to their respective original licenses and attribution requirements. Original sources are referenced through the metadata provided in this repository. The manuscripts and human-written reviews are reproduced without modification.

This dataset is provided to support scientific reproducibility and further research. Redistribution of third-party content does not imply endorsement of this repository or its findings by the original authors, reviewers, conference organizers, publishers, or hosting platforms.

If you are a rights holder and believe that material in this repository has been included in error or that its attribution or licensing information is incorrect, please contact us through the repository's contact information so that we can review and address the concern.
