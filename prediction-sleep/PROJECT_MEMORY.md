## Project Memory — Next-Day Fatigue Prediction (Apple Watch)

**Location**: `/Users/shashikant/Desktop/GMU_HW/Cs655_Project/Prediction sleep`

### Mobile-First Memory (Current)
- Use `MOBILE_READINESS_MEMORY.md` as the primary living memory for mobile-computing constraints, deployment readiness, and iteration logs.

### Status at Handoff (Feb 8, 2026)
- Apple Health export parsed; night-level dataset built; baseline models trained.
- Best current model: Random Forest (`models/rf.pkl`) with test F1 ≈ 0.51, accuracy ≈ 0.52 on 108 held-out nights (balanced labels 0/1).
- Streamlit demo and CLI predictor exist; both point to the RF model.

### Key Artifacts
- Parsed tables: `parsed_tables/{sleep,hr,hrv,resp,spo2}.csv`
- Engineered features: `dataset/night_features.csv` (359 nights, includes medians/stds for HR/HRV/resp/SpO2)
- Labels: `dataset/fatigue_labels.csv` (user provided; balanced 180/179)
- Merged/model data: `dataset/model_data.csv` (label_date = night_date + 1 day)
- Splits: `dataset/train.csv`, `dataset/test.csv` (70/30 time-based)
- Models: `models/rf.pkl` (best), `models/logreg.pkl`
- Reports: `reports/metrics.json`, `reports/confusion_matrix.png`, `reports/feature_importance.png`
- Apps/Scripts:
  - `scripts/step1_parse_health.py` — parse export.xml to CSVs
  - `scripts/step2_build_features.py` — build night features (current night window: shift start by 6h; adds mean/min/max/median/std physio stats)
  - `scripts/step3_merge_labels.py` — align fatigue labels (creates template if missing)
  - `scripts/step4_time_split.py` — time-aware split
  - `scripts/step5_train_models.py` — trains LogReg & RF, saves metrics/plots
  - `predict.py` — CLI prediction (uses RF)
  - `app/streamlit_app.py` — simple UI (uses RF)

### How to Reproduce Current State
```bash
cd "/Users/shashikant/Desktop/GMU_HW/Cs655_Project/Prediction sleep"
source .venv/bin/activate
# rebuild features → merge labels → split → train
.venv/bin/python scripts/step2_build_features.py --parsed-dir parsed_tables --out dataset/night_features.csv
.venv/bin/python scripts/step3_merge_labels.py
.venv/bin/python scripts/step4_time_split.py
.venv/bin/python scripts/step5_train_models.py
# predict / app
./predict.py --date YYYY-MM-DD
streamlit run app/streamlit_app.py
```

### Gaps / Next Improvement Ideas
1) Night window: redefine to 8 PM–10 AM or use actual bedtime/wake markers; rerun steps 2–5.
2) Trend features: rolling 3- and 7-day means/deltas for sleep duration, HRV, HR, resp.
3) Daytime context: add next-day daytime HR/HRV/resp/steps averages as features (steps likely in export).
4) Model tuning: RF hyperparameters, Gradient Boosting/XGBoost; time-series cross-validation.
5) Label quality: confirm consistency of fatigue labels and their timing relative to sleep night.

### Notes
- Dataset is small; expect modest ceilings unless richer features are added.
- Time split is deterministic; reruns will give the same metrics unless features/models change.
