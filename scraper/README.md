# Survey data pipeline

Scripts that build the venue lists and figures for the survey on AI-assisted peer review policies. The final datasets (`conference_db.csv`, `medical_journals.csv`) are already included with policies filled in, you only need to re-run if you want to regenerate from scratch or update the data.

## Setup

```
pip install requests matplotlib numpy
```

## To regenerate the datasets

```bash
python build_conference_db.py   
python build_medical_journals.py 
python make_survey_figures.py   
```

The JCR exports in `medical_jcr_exports/` were downloaded from jcr.clarivate.com


## What each script does

**`build_conference_db.py`** downloads the CORE 2023 rankings export, filters to A*/A venues in AI/NLP fields (FOR codes 4602, 4611, 4603, 4605), and looks up each conference website via WikiCFP. Outputs `conference_db.csv`.

**`build_medical_journals.py`**  reads the 6 Clarivate JCR exports from
`medical_jcr_exports/`, takes the top 8 journals per subcategory by 2024 JIF, deduplicates by ISSN. Outputs `medical_journals.csv`.

**`make_survey_figures.py`** generates the two bar charts for the results chapter.
Edit the counts at the top if the classification changes, then re-run.
