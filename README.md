# CS655 Mobile Collaboration Repo

This folder is a GitHub-ready collaboration copy of the project.

## Included

- `prediction-sleep/`
  - Data pipeline scripts (`scripts/`)
  - Parsed/engineered datasets (`parsed_tables/`, `dataset/`)
  - Trained baseline models (`models/`)
  - Reports and metrics (`reports/`)
  - App and CLI (`app/`, `predict.py`)
  - Project memory docs, including mobile-readiness memory
- `docs/`
  - `CS655_Project_final.pdf`
  - `Project_Proposal.docx`
  - `Reference_Papers_for_Project.md`

## Excluded by Default

- Local virtual environment (`.venv/`)
- Raw Apple Health export blobs (large/private), e.g. `export.xml`, `export_cda.xml`

## Quick Start

```bash
cd prediction-sleep
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

All runtime commands below should be executed after activating `.venv`.

Run pipeline:

```bash
python scripts/step2_build_features.py --parsed-dir parsed_tables --out dataset/night_features.csv
python scripts/step3_merge_labels.py
python scripts/step4_time_split.py
python scripts/step5_train_models.py
```

Run prediction app:

```bash
streamlit run app/streamlit_app.py
```

## Push to GitHub

From this folder (`CS655_Mobile_Collab`):

```bash
git add .
git commit -m "Initial collaboration snapshot"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
