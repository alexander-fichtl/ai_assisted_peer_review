# Overlap Pipeline

Measures how much AI peer reviews overlap with a pool of human peer reviews of the same
paper. Each review is decomposed into individual concerns (Stage 1), AI concerns
are matched to human concerns (Stage 2), and overlap metrics recall-style
coverage, precision, additional findings are computed (Stage 3), with figures
(Stage 4).

## Setup

1. `pip install openai python-dotenv pandas matplotlib`
2. Put `OPENAI_API_KEY=sk-...` in `.env` (root)
3. `pip install tiktoken` (exact paper-token counts) and `pip install sentence-transformers` (semantic baseline matcher in Stage 3b)

## Run, in order

```bash
# 0. One-time: unzip the papers text.
python extract_processed_pdfs.py

# 1. Check data loads (just to see if data ok, doesn't do anything, functions are used later)
python data_loaders.py

# 2. Stage 1 decompose reviews into concerns.
python run_full_extraction.py

# 3. Stage 2 match AI<->human + human<->human leave-one-out.
python run_full_matching.py

#4. Stage 3 compute metrics + tables
python score_overlap.py

# 5 Stage 4 render figures
python plot_results.py

# 6. Stage 3b non-LLM baseline matchers
python baseline_overlap.py
```

## What comes out

```
outputs/
  concerns/<dataset>/<paper_id>/{gpt,llama,qwen,human_*}.json   Stage 1
  matches/<dataset>/<paper_id>/{gpt,llama,qwen,h2h_*}.json      Stage 2
  scores/
    per_pair.csv         one row per (paper, source): every metric + counts
    matched_pairs.csv    one row per matched pair (manual inspection)
    aggregate.csv / .md  coverage by source / dataset / seriousness / type / accept-reject
    baseline_*.csv/.md   Stage 3b non-LLM matcher comparison
  plots/fig_*.{pdf,png}  Stage 4 figures
```

## Set number of papers to check
To run on all papers, set `N_ICLR_PAPERS = None` and `N_NATURE_PAPERS = None` at the top of both `run_full_extraction.py` and `run_full_matching.py`. 