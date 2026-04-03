# Pipeline Working Explanation

Last updated: 2026-04-03 (America/New_York)

## Goal

Build a fatigue prediction workflow that:
- learns robust wearable representations from public data,
- fine-tunes on project fatigue labels,
- distills to a mobile-size model,
- validates robustness for research reporting.

## End-to-End Flow

1. **Phase 6: Public Pretraining (`step6_public_pretrain.py`)**
- Inputs:
  - SleepAccel root
  - PPG-DaLiA root (extracted field-study folder)
- What happens:
  - Load/public-parse datasets
  - Standardize to window-level numeric table
  - Fit compact encoder: `SimpleImputer -> StandardScaler -> PCA`
- Outputs:
  - `artifacts/public_pretrain/public_encoder.pkl`
  - `artifacts/public_pretrain/public_feature_space.json`
  - `artifacts/public_pretrain/public_pretrain_report.json`
  - optional `public_window_table.csv`

2. **Phase 7: Transfer Finetuning (`step7_transfer_finetune.py`)**
- Inputs:
  - `dataset/model_data.csv`
  - `artifacts/public_pretrain/public_encoder.pkl`
- What happens:
  - Build ablation spaces: baseline / transfer / combined
  - Time-aware CV (`TimeSeriesSplit`)
  - Train candidate models and select champion
- Outputs:
  - `reports/public_transfer_ablation.csv`
  - `reports/public_transfer_report.json`
  - `models/transfer_champion.pkl`

3. **Phase 8: Distillation (`step8_distill_mobile.py`)**
- Inputs:
  - teacher model (`transfer_champion.pkl` or fallback `mobile_champion.pkl`)
  - `dataset/train.csv`, `dataset/test.csv`
- What happens:
  - Train compact student candidates
  - Evaluate size + latency + weighted F1
  - Calibrate decision threshold with out-of-fold time-split probabilities, with safety fallback to `0.5` when unstable
  - Export champion + linear contract
- Outputs:
  - `models/distilled_mobile.pkl`
  - `reports/distillation_report.json`
  - `reports/distillation_scores.csv`
  - `artifacts/distilled_linear_contract.json` (if linear champion, includes `decision_threshold`)

4. **Phase 9: Research Evaluation (`step9_research_eval.py`)**
- Inputs:
  - distilled model (or fallback model)
  - `dataset/test.csv`
- What happens:
  - Clean weighted F1
  - bootstrap CI
  - synthetic missingness/noise sweeps
- Outputs:
  - `reports/robustness_metrics.json`
  - `reports/research_summary.md`

## Correct Dataset Paths

SleepAccel:
- `prediction-sleep/dataset/public/SleepAccel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0`

PPG-DaLiA:
- Download package includes an inner `data.zip`
- Extract that inner archive first
- Use extracted root:
  - `prediction-sleep/dataset/public/PPG-DaLiA/ppg+dalia/extracted/PPG_FieldStudy`

## One-Command Execution

```bash
cd prediction-sleep
scripts/run_phases_6_9.sh \
  --sleepaccel-root dataset/public/SleepAccel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0 \
  --ppg-dalia-root dataset/public/PPG-DaLiA/ppg+dalia/extracted/PPG_FieldStudy \
  --ppg-max-subjects 3 \
  --skip-latency
```

Logs:
- `logs/phases_6_9/<timestamp>/`

## Why This Is Research-Ready

- Public-data pretraining improves representation robustness.
- Time-aware validation avoids leakage from temporal drift.
- Distillation enforces mobile deployment constraints.
- Bootstrap + corruption sweeps support stronger scientific claims.

## Current Verified Snapshot

- Public pretraining loaded both datasets:
  - `sleepaccel`: 25,971 rows / 31 subjects
  - `ppg_dalia`: 3,021,415 rows / 15 subjects
  - combined: 3,047,386 rows / 46 subjects
- Transfer stage reports `encoder_available: true` with `public_enc_00..07` features.
- Distilled mobile champion: `logreg_distilled_c2_0`.
