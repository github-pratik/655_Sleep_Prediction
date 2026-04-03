# Phased Execution Runbook

Last updated: 2026-04-03 (America/New_York)

## 1) How to know which agent is finished

Use `AGENT_STATUS.md` as the single source of truth:
- `RUNNING`: still executing
- `COMPLETED`: delivered expected outputs
- `BLOCKED`: needs intervention

Current worker mapping:
- `Galileo`: `step6` + `src/public_data`
- `Sagan`: `step7` + `src/transfer`
- `Hooke`: `step8/step9` + benchmark template

Runtime confirmation in this workspace:
- All three are `COMPLETED` (as of 2026-04-03).

Quick completion check by artifacts:
- `artifacts/public_pretrain/public_encoder.pkl`
- `reports/public_transfer_report.json`
- `reports/distillation_report.json`
- `reports/robustness_metrics.json`

## 2) End-to-end command sequence

Run from:
`/Users/shashikant/Desktop/GMU_HW/Cs655_Project/CS655_Mobile_Collab/prediction-sleep`

### One-command runner (phases 6 to 9 + logs)

Recommended local layout:
`prediction-sleep/dataset/public/SleepAccel`
`prediction-sleep/dataset/public/PPG-DaLiA`

```bash
chmod +x scripts/run_phases_6_9.sh
scripts/run_phases_6_9.sh \
  --sleepaccel-root dataset/public/SleepAccel \
  --ppg-dalia-root dataset/public/PPG-DaLiA \
  --ppg-max-subjects 3 \
  --skip-latency
```

If your datasets are outside the repo, pass absolute paths instead:
```bash
scripts/run_phases_6_9.sh \
  --sleepaccel-root /absolute/path/to/SleepAccel \
  --ppg-dalia-root /absolute/path/to/PPG-DaLiA \
  --skip-latency
```

No public dataset yet (skip step6, still run 7 to 9):
```bash
scripts/run_phases_6_9.sh --skip-latency
```

Logs are written to:
`logs/phases_6_9/<timestamp>/`

### Archive or remove unwanted run files

Create a zip archive of logs + redundant PPG `data.zip`:
```bash
scripts/archive_cleanup.sh
```

Archive and then delete those files:
```bash
scripts/archive_cleanup.sh --delete
```

### Phase 6: public pretraining

With real public datasets:
```bash
.venv/bin/python scripts/step6_public_pretrain.py \
  --sleepaccel-root dataset/public/SleepAccel \
  --ppg-dalia-root dataset/public/PPG-DaLiA \
  --output-dir artifacts/public_pretrain \
  --save-window-table
```

If you have no public dataset yet:
```bash
.venv/bin/python scripts/step6_public_pretrain.py --help
```

### Phase 7: transfer finetuning

```bash
.venv/bin/python scripts/step7_transfer_finetune.py \
  --model-data dataset/model_data.csv \
  --encoder artifacts/public_pretrain/public_encoder.pkl
```

Fast rerun (no latency benchmark):
```bash
.venv/bin/python scripts/step7_transfer_finetune.py \
  --skip-latency \
  --encoder artifacts/public_pretrain/public_encoder.pkl
```

### Phase 8: distill to mobile

```bash
.venv/bin/python scripts/step8_distill_mobile.py --skip-plots
```

### Phase 9: research evaluation

```bash
.venv/bin/python scripts/step9_research_eval.py --bootstrap 500
```

## 3) Expected outputs by phase

Phase 6:
- `artifacts/public_pretrain/public_encoder.pkl`
- `artifacts/public_pretrain/public_feature_space.json`
- `artifacts/public_pretrain/public_pretrain_report.json`

Phase 7:
- `reports/public_transfer_ablation.csv`
- `reports/public_transfer_report.json`
- `models/transfer_champion.pkl`

Phase 8:
- `reports/distillation_report.json`
- `reports/distillation_scores.csv`
- `models/distilled_mobile.pkl`
- `artifacts/distilled_linear_contract.json` (for linear champion, includes calibrated `decision_threshold`)

Phase 9:
- `reports/robustness_metrics.json`
- `reports/research_summary.md`

## 4) Latest validated run snapshot

- Transfer champion (step7): `transfer / hist_gb`
- Distilled champion (step8): `logreg_distilled_c2_0`
- Distilled model size: ~4.95 KB
- Distilled p95 single-pred latency (desktop proxy): ~4-6 ms
- Clean weighted F1 (step9): `0.4815`
- Bootstrap 95% CI (weighted F1): `0.3814` to `0.5783`

## 5) Notes

- `step8` auto-falls back to `models/mobile_champion.pkl` if the transfer teacher has zero feature overlap with train/test splits.
- `step8` calibrates threshold using out-of-fold train probabilities (time-aware splits), then applies a safety gate; if gain/shift is unstable it falls back to `0.5` before writing the distilled linear contract.
- You may see sklearn version warnings when loading older pickles; retraining teacher artifacts in the current environment removes this.
