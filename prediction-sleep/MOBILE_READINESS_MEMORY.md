# Mobile Readiness Memory

Last updated: 2026-04-02 (America/New_York)
Project: Next-Day Fatigue Prediction from Apple Watch data
Primary codebase: `Prediction sleep/`

## Purpose

This is the living memory document for mobile readiness.
Use it to track:
- Current deployability status for on-device inference
- Hard constraints (size, latency, memory, battery)
- What changed each iteration and why
- Remaining blockers before a mobile-computing quality demo

## Mobile Constraint Targets

- On-device inference only (no cloud dependency at prediction time)
- Model size: <= 500 KB (watch target), <= 1 MB (phone target)
- Single prediction latency: <= 250 ms (watch), <= 100 ms (phone)
- Peak memory during inference: <= 25 MB (watch), <= 80 MB (phone)
- Robustness under missing sensor data: graceful degradation, no crashes

## Current Baseline Snapshot

- Data rows: 359 labeled nights
- Test rows: 108
- Current best test model: Random Forest
  - weighted F1: 0.511
  - accuracy: 0.519
  - source: `reports/metrics.json`

### Model Artifact and Latency Baseline (desktop proxy, local benchmark)

- `models/logreg.pkl`
  - size: 5.03 KB
  - p50 single-prediction latency: 0.964 ms
  - p95 single-prediction latency: 1.03 ms
- `models/rf.pkl`
  - size: 2967.35 KB
  - p50 single-prediction latency: 14.799 ms
  - p95 single-prediction latency: 27.037 ms

Notes:
- These are local desktop proxy numbers, not device measurements.
- Relative ranking is still useful for mobile planning.

## Mobile Readiness Scorecard

Status scale: `READY`, `PARTIAL`, `NOT READY`

1. Model footprint for watch-class deployment: `PARTIAL`
- Logistic Regression meets size budget.
- Random Forest exceeds preferred watch budget (~2.9 MB).

2. Inference speed budget likelihood: `PARTIAL`
- Likely fine for phone.
- Watch estimate unknown until real CoreML/TFLite benchmark.

3. Native mobile runtime compatibility: `NOT READY`
- Current inference depends on Python + `joblib` + scikit-learn pipeline.
- No CoreML/TFLite/ONNX artifact exists yet.

4. On-device feature computation feasibility: `PARTIAL`
- Current features are low-order statistics (good).
- Pipeline currently implemented in pandas scripts, not native/mobile code.

5. Robustness to mobile sensor gaps/noise: `PARTIAL`
- Median imputation exists in training pipeline.
- No explicit stress-test curves under synthetic dropout/noise yet.

6. Energy and memory profiling on target device: `NOT READY`
- No measured battery or peak-memory runs on phone/watch.

7. Mobile app integration path: `NOT READY`
- Existing UI is Streamlit, not an iOS/Android/watch runtime path.

## Key Gaps to Close First

1. Replace Python-specific serving path with mobile-native model artifact.
2. Add real mobile benchmark harness (latency, memory, battery).
3. Add missing-data robustness evaluation and training-time mitigation.
4. Keep model within size budget while preserving/raising F1.

## Implementation Plan (Mobile-First)

### Phase 1: Mobile Inference Contract (immediate)

1. Freeze a strict feature schema and ordering.
2. Export preprocessing parameters (imputer medians, scaler means/stds) to JSON.
3. Add deterministic parity tests:
   - Python pipeline prediction vs exported-runtime prediction on same inputs.

Deliverables:
- `artifacts/feature_schema.json`
- `artifacts/preprocess_params.json`
- `reports/parity_check.json`

### Phase 2: Lightweight Model Track

1. Optimize Logistic Regression baseline for mobile:
   - class weights
   - threshold tuning for weighted F1 / macro-F1
2. Keep Random Forest as accuracy reference only.
3. Add compact candidate (small tree ensemble) only if within size budget.

Deliverables:
- Updated `reports/metrics.json` with mobile-eligible model highlighted
- `reports/model_size_latency_table.csv`

### Phase 3: Robustness Under Mobile Sensing

1. Inject synthetic missingness/noise on validation data.
2. Plot performance vs corruption level (0%, 10%, 30%, 50%).
3. Introduce low-cost robustness strategy if needed:
   - feature masking indicators
   - stronger imputation policy

Deliverables:
- `reports/robustness_curves.png`
- `reports/robustness_metrics.json`

### Phase 4: Device-Level Validation

1. Export mobile artifact (CoreML for Apple path; TFLite for cross-platform path).
2. Measure actual on-device latency/memory/battery.
3. Capture tradeoff table: accuracy vs latency vs size vs energy.

Deliverables:
- `reports/mobile_benchmark.md`
- `reports/mobile_tradeoff_table.csv`

## Decision Log

2026-04-02:
- Decision: treat this as a mobile-computing project with on-device inference as a hard requirement.
- Decision: maintain two tracks:
  - accuracy reference model (RF)
  - mobile-eligible model (LogReg-first)

## Iteration Log Template

Append entries below each time work is completed.

```
Date:
Change summary:
Files touched:
Model(s):
Metrics (accuracy/F1):
Size (KB/MB):
Latency (p50/p95):
Memory/Battery notes:
Open issues:
Next step:
```

## Iteration Log

Date: 2026-04-02
Change summary: Added a mobile-first model training/evaluation track and created a React Native edge-inference demo scaffold.
Files touched:
- `scripts/step5_train_mobile_models.py`
- `predict.py`
- `app/streamlit_app.py`
- `artifacts/feature_schema.json`
- `artifacts/mobile_linear_contract.json`
- `reports/mobile_model_scores.csv`
- `reports/mobile_model_report.json`
- `models/mobile_champion.pkl`
- `models/mobile_champion_name.txt`
- `mobile_demo_app/*`
Model(s):
- Champion (accuracy-oriented mobile-eligible): `rf_small_n96_d8`
- Best linear app contract model: `logreg_c0_1`
Metrics (accuracy/F1):
- Champion test accuracy: `0.519`
- Champion weighted F1: `0.517`
Size (KB/MB):
- Champion model size: `~325.3 KB`
- Best linear model size: `~5.0 KB`
Latency (p50/p95):
- Champion single inference: `p50 ~14.7 ms`, `p95 ~15.3 ms`
- Best linear single inference: `p50 ~0.9 ms`, `p95 ~1.0 ms`
Memory/Battery notes:
- No on-device battery/memory benchmark yet; current timings are desktop proxy.
Open issues:
- Need parity-check report between sklearn and linear contract inference.
- Need real phone/watch benchmark run.
Next step:
- Build a mobile benchmark harness and add parity validation script.
