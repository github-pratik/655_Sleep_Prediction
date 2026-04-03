# CS655 Mobile Collaboration Repo

[![Prediction Sleep Tests](https://github.com/github-pratik/655_Sleep_Prediction/actions/workflows/prediction-sleep-tests.yml/badge.svg)](https://github.com/github-pratik/655_Sleep_Prediction/actions/workflows/prediction-sleep-tests.yml)

This folder is a GitHub-ready collaboration copy of the project.

## Included

- `prediction-sleep/`
  - Data pipeline scripts (`scripts/`)
  - Parsed/engineered datasets (`parsed_tables/`, `dataset/`)
  - Trained baseline models (`models/`)
  - Reports and metrics (`reports/`)
  - Implementation references (`docs/references/`)
  - Reusable local skills (`skills/`)
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

## Public Datasets (Manual Download Required)

Yes, download SleepAccel and PPG-DaLiA manually first.
This repo does not auto-download public datasets because dataset terms and folder layouts can vary.

Recommended local layout:

```bash
prediction-sleep/data/public/SleepAccel/
prediction-sleep/data/public/PPG-DaLiA/
```

Then run the phased pipeline with explicit paths:

```bash
cd prediction-sleep
chmod +x scripts/run_phases_6_9.sh
scripts/run_phases_6_9.sh \
  --sleepaccel-root data/public/SleepAccel \
  --ppg-dalia-root data/public/PPG-DaLiA \
  --skip-latency
```

If you do not have public datasets yet, you can still run phases 7 to 9:

```bash
scripts/run_phases_6_9.sh --skip-latency
```

Useful references:
- [SleepAccel (PhysioNet)](https://physionet.org/content/sleep-accel/)
- [PPG-DaLiA (UCI)](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)

## Train Mobile-First Models

```bash
python scripts/step5_train_mobile_models.py --skip-plots
```

Key outputs:
- `models/mobile_champion.pkl`
- `reports/mobile_model_report.json`
- `reports/mobile_model_scores.csv`
- `artifacts/mobile_linear_contract.json`

## Rebuild Full Data Pipeline (Optional)

```bash
python scripts/step2_build_features.py --parsed-dir parsed_tables --out dataset/night_features.csv
python scripts/step3_merge_labels.py
python scripts/step4_time_split.py
python scripts/step5_train_mobile_models.py --skip-plots
```

## Research Pipeline (Phases 6 to 9)

Single-command runner:

```bash
cd prediction-sleep
scripts/run_phases_6_9.sh --skip-latency
```

Detailed phase docs:
- `prediction-sleep/docs/PHASED_EXECUTION_RUNBOOK.md`
- `prediction-sleep/PUBLIC_DATA_MOBILE_RESEARCH_PLAN.md`

Reproducible environment lock:
- `prediction-sleep/requirements.lock.txt`

## Run Prediction App (Desktop)

```bash
streamlit run app/streamlit_app.py
```

## iPhone HealthKit Demo (Native iOS)

Path:
- `prediction-sleep/ios_healthkit_demo`

Generate/open Xcode project:

```bash
cd prediction-sleep/ios_healthkit_demo
./generate_project.sh
```

Open this in Xcode:
- `prediction-sleep/ios_healthkit_demo/SleepFatigueEdgeDemo.xcodeproj`

### Simulator Testing

Simulator cannot fetch real Apple Health/Watch data, but you can test full app flow:
1. Run app in Simulator.
2. Tap `Load Demo Data (Simulator Safe)`.
3. Tap `Run On-Device Prediction`.

For real HealthKit fetch from Apple Watch data, run on a physical iPhone.

## Push to GitHub

From this folder (`CS655_Mobile_Collab`):

```bash
git add .
git commit -m "Initial collaboration snapshot"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
