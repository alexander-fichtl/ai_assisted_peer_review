# lm-based-academic-review-support-sc

Code and supplementary material for LLM-based academic peer review generation and evaluation. Covers two venues: **ICLR** and **Nature Communications**.

---

## Repository Structure

```
ai_assisted_peer_review/
├── data/                     # Data folder
├── data_generation/          # Fetching papers and peer reviews
│   ├── ICLR/                 # ICLR pipeline via OpenReview API
│   ├── natureCom/            # Nature Communications pipeline via Research Square
│   ├── pdf_fetcher_docling.py
│   └── extract_formulas_for_md.py
├── overlap_pipeline/         # Code for the overlap_pipeline and evaluation
├── review_generation/        # LLM review generation
│   ├── deepseek_llava/       # DeepSeek-R1-Distill-Qwen-32B on LRZ
│   ├── llama/                # Llama-4-Scout-17B on RunPod
│   ├── llama_LRZ/            # Llama-4-Scout-17B on LRZ (sbatch)
│   ├── qwen/                 # Qwen3-VL-235B on RunPod
│   └── gpt/                  # GPT-5 via API
└── review_evaluation/        # Comparing LLM vs. human reviews
    ├── granuscore/           # GranuScore metric
    ├── llm-as-a-judge/       # GPT-based quality scoring
    ├── score-comparison/     # Given-score visualizations
    └── pairwise-manual/      # Streamlit human preference app
└── paper_supplementary_material/        # Supplementary material mentioned in the paper
```
---

## data_generation

### Shared tools
- `pdf_fetcher_docling.py`: Downloads a PDF from a URL and preprocesses it with [Docling](https://github.com/docling-project) into `.txt`, `.md`, and `.json` files. Outputs are saved to `preprocessed_pdfs/<pdf_name>/`.
- `extract_formulas_for_md.py`: Post-processes Docling Markdown output to restore LaTeX formulas.

---

### ICLR
Collects submissions and their peer reviews from [OpenReview](https://openreview.net/) v2.

1. **`openreview_api.py`**: Fetches all public submissions for a given venue (e.g., `ICLR.cc/2025/Conference`) via the [OpenReview API v2](https://docs.openreview.net/reference/api-v2).
   - Stores one `<id>.json` per paper under `data/openreview_basesets/<VENUE_ID>/submissions_jsons/`.
   - Also writes a summary `submissions.csv` and `submissions.jsonl`.
   - Helper functions (`get()`, `get_reviews()`, etc.) operate on the loaded JSON objects.

2. **`dataset_generator_ICLR.py`**: Builds structured datasets from the fetched submissions.
   - Reads all paper IDs for a venue (accepted and rejected).
   - Optionally samples `n` random papers and saves a `dataset.csv` with meta information to `data/final_datasets/<dataset_name>/`.

3. **`create_index.py`**: Creates or updates the index file for a dataset. Only necessary for self-defined sets.

---

### Nature Communications
Collects papers from [Research Square](https://www.researchsquare.com/) and their peer reviews from [Nature Communications](https://www.nature.com/ncomms/).

1. **`paper-id-set_selector.py`**
   - Uses the [Research Square API](https://www.researchsquare.com/developers/docs/) to fetch papers matching defined search criteria and saves the result to `data/papersets_ids/natureCom/pool_<setname>.json`.
   - Also selects a random sample of `samplesize` papers from a pool and saves it to `data/papersets_ids/natureCom/sample_<setname>_<samplesize>.json`. Papers can be banned; gaps are filled automatically on rerun.
   - Creates a named set entry (`NatureCom_<setname>_<samplesize>`) in `data/papersets_ids/natureCom/index.json`.

2. **Manual step**: The mapping from Research Square paper IDs to Nature Communications IDs, plus links to papers and reviews, must be added manually to `data/papersets_ids/natureCom/index.json`.

3. **`pdf_fetcher_and_processer.py`**: Downloads paper PDFs from Research Square and review files from Nature Communications using links in the index. Calls `pdf_fetcher_docling` to produce structured JSON.

4. **`reviews_extractor.py`**: Extracts the first round of peer reviews from the fetched JSON into a structured format.
   > **Note:** Nature Communications reviews often have irregular formatting. Check results manually.

---

## review_generation
Each subfolder targets a specific model and compute environment. All reviewers consume paper files preprocessed by Docling (`.json`/`.md` + images) and write output reviews to a `reviews/` directory.

### deepseek_llava — LRZ interactive nodes
Runs [`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) via vLLM. Uses [LLaVA](https://llava-vl.github.io/) to generate image descriptions.

- `deepseek_llava.py`: End-to-end: PDF → prompt → review. (But to avoid having to run both models at the same time; use the split scripts below.)
- `pdf_to_prompt.py`: Converts a PDF to a text prompt.
- `prompt_to_review.py`: Generates a review from a prompt.
- `prompt_to_review_CutPrompt.py`: Same as above but truncates input after 14,000 tokens and does not carry over previous responses (workaround for the 16,384 token limit).
- `lrz_instructions.txt`: Step-by-step LRZ setup. Replace `xx00xxx2` with your LRZ ID.
- `build_deepseek14_env.sh`: Environment setup script.

### llama — RunPod
Runs [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) via Transformers on RunPod.

- `ICLR/reviewer_llama_ICLR.py` / `natureCom/reviewer_llama_natureCom.py`: Takes a Docling `.md` file and images and writes a review to `reviews/<paper_name>/`.
- `ICLR/runpod_commands.txt` / `natureCom/runpod_commands.txt`: Commands for launching and interacting with the RunPod instance.

### llama_LRZ — LRZ sbatch jobs
Runs [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) via Transformers on LRZ.

- `ICLR/reviewer_llama_ICLR.py` / `natureCom/reviewer_llama_natureCom.py`: Takes a Docling `.md` file and images and writes a review to `reviews/<paper_name>/`.
- `ICLR/run_review_llama_ICLR.sbatch` / `natureCom/run_review_llama_natureCom.sbatch`: Submits the reviewer as a SLURM sbatch job on LRZ.
- `lrz_instructions_sbatch.txt`: Step-by-step sbatch setup. Replace `xx00xxx2` with your LRZ ID.
- `build_llama_env.sh`: Environment setup script.

### qwen — RunPod
Runs [`Qwen/Qwen3-VL-235B-A22B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking) via vLLM.

- `ICLR/reviewer_qwen_ICLR.py` / `natureCom/reviewer_qwen_natureCom.py`: Takes a Docling `.md` file and images and writes a review to `reviews/<paper_name>/`.
- `runpod_commands.txt` — Commands for launching and interacting with the RunPod instance.

### gpt — OpenAI API
Runs `gpt-5` via the OpenAI API. 

- `ICLR/reviewer_gpt_ICLR.py`: Takes a Docling `.md` file and images; answers the standard ICLR review questions one by one. Writes output to `data/reviews_generated/ICLR/gpt/<paper_id>/`.
- `natureCom/reviewer_gpt_natureCom.py`: Same for Nature Communications. Writes output to `data/reviews_generated/natureCom/gpt/<paper_id>/`.
- `chatgpt/`: Outdated. Prompts and info for manual review generation via the ChatGPT web interface (not used by the API scripts above).

---

## review_evaluation
Four complementary evaluation methods for comparing LLM-generated reviews against human reviews.

### granuscore

Computes [GranuScore](add reference) metrics on generated and human reviews.

- `evaluator_granuscore.py`: Loads reviews, scores them with GranuScore, filters, aggregates, and plots results. Entry point: `run_comparison()`.
- `run_ICLR.py` / `run_natureCom.py`: Runner scripts for each venue.

### llm-as-a-judge
Uses GPT (via OpenAI API) to score review quality across five dimensions (**Understanding**, **Coverage**, **Evidence Support**, **Constructiveness**, **Conciseness**) on a 1–5 Likert scale, plus an overall score from 1–7.

- `evaluator_llmjudge.py`: Sends `paper + review` to GPT and writes results to CSV and JSON. Entry point: `judge_reviews()`.
  - Requires `OPENAI_API_KEY` environment variable.
- `run_ICLR.py` / `run_natureCom.py`: Runner scripts that load reviews into a flat list and call `judge_reviews()`.
- `visualizeJudgescores.py`: Plots the judge scores.

### score-comparison
Visualizes the *given* reviewer scores (from OpenReview / NatureCom) per paper, overlaid with LLM-generated review scores.

- `compareReviews_givenscorePerPaper.py`: Loads `dataset.csv` and generated reviews, produces scatter/box plots per paper. 
    - Configurable: which LLM sources to include, whether to show human IQR/median, accept/reject shading, and score axis range.

### pairwise-manual
A [Streamlit](https://streamlit.io/) web app for human blind pairwise preference evaluation.

- `manual_pairwise_review_app.py`: Displays a paper link and two anonymized reviews side-by-side; collects preferences and exports results to a per-user CSV. Reloads previous sessions to avoid showing the same pair twice.

  **Run:**
  ```bash
  cd <project_root>
  python -m streamlit run review_evaluation/pairwise-manual/manual_pairwise_review_app.py
  ```

  Configure `ID_user`, `VAR_reviewers`, and `DATASET_NAME` at the top of the file before running.

---

## Environment

Set the following environment variables (e.g., in `.env`):

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | `llm-as-a-judge` evaluator, `gpt` reviewer |
| `HF_TOKEN` | `llama`, `deepseek_llava` (LRZ deployments) |

For LRZ deployments, NVIDIA NGC credentials are also required — see the respective `lrz_instructions*.txt` files.
