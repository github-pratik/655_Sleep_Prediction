# Validation Checklist (Phases 6 to 9)

Use this checklist after each run to confirm technical correctness.

## Phase 6: Public Pretraining

- Command completes without `No public datasets were loaded`.
- Files exist:
  - `artifacts/public_pretrain/public_encoder.pkl`
  - `artifacts/public_pretrain/public_feature_space.json`
  - `artifacts/public_pretrain/public_pretrain_report.json`
- `public_pretrain_report.json` has non-zero rows and subjects.

Quick check:
```bash
jq '.table.rows, .table.subjects' artifacts/public_pretrain/public_pretrain_report.json
```

## Phase 7: Transfer Finetuning

- Files exist:
  - `reports/public_transfer_ablation.csv`
  - `reports/public_transfer_report.json`
  - `models/transfer_champion.pkl`
- `cv_splits_used >= 3`.

Quick check:
```bash
jq '.summary.cv_splits_used, .summary.champion_ablation, .summary.champion_model' reports/public_transfer_report.json
```

## Phase 8: Distillation

- Files exist:
  - `reports/distillation_report.json`
  - `reports/distillation_scores.csv`
  - `models/distilled_mobile.pkl`
- `num_mobile_eligible >= 1`.

Quick check:
```bash
jq '.summary.champion, .summary.num_mobile_eligible' reports/distillation_report.json
```

## Phase 9: Research Eval

- Files exist:
  - `reports/robustness_metrics.json`
  - `reports/research_summary.md`
- `clean_weighted_f1` is present.
- Bootstrap confidence interval fields are present.

Quick check:
```bash
jq '.clean_weighted_f1 // .clean_weighted_f1, .bootstrap.ci95_low, .bootstrap.ci95_high' reports/robustness_metrics.json
```

## Fast Failure Diagnostics

- If Phase 6 fails:
  - Check dataset folder names include `motion`, `heart_rate`, `labels`.
  - Confirm path passed to `--sleepaccel-root` points to extracted dataset root.
- If Phase 7/8 fails:
  - Ensure `dataset/model_data.csv`, `dataset/train.csv`, and `dataset/test.csv` exist.
  - Re-run with `--skip-latency` for faster debug loops.
- If pickle compatibility warnings appear:
  - Warnings are expected across sklearn versions; run completes with compatibility shim.
