# Public Data + Mobile Compute Research Plan

Last updated: 2026-04-03 (America/New_York)
Project: Next-Day Fatigue Prediction from Apple Watch data

## Short Answer

Yes. You can use publicly available datasets to improve generalization while still deploying a mobile-class model.
The best approach is multi-stage training:
- learn robust wearable representations from public data
- fine-tune and calibrate on your project-specific fatigue labels
- distill to a compact on-device model (phone/watch budget)

## Execution Phases

1. Data ingestion
- Build local loaders for SleepAccel and PPG-DaLiA.
- Standardize every archive into a window-level table.

2. Public pretraining
- Fit a compact encoder on public window features.
- Save a portable artifact that can be reused offline.

3. Fine-tuning and transfer
- Combine public embeddings with the project fatigue dataset.
- Tune for time-aware validation, not random split accuracy.

4. Mobile packaging
- Keep a lightweight edge model for iPhone/watch deployment.
- Export a contract or CoreML-compatible artifact.

## Parallel Workstreams

- Loader workstream: archive discovery, file parsing, window table creation.
- Modeling workstream: encoder training, ablations, distillation, calibration.
- Evaluation workstream: walk-forward CV, robustness curves, mobile benchmarks.

## Why This Helps Here

Current project data is small (359 labeled nights) and has major missingness.
Public wearable datasets can improve:
- sensor-noise robustness
- feature extractor quality
- training stability

Then your private fatigue labels keep the task personalized and clinically relevant.

## Public Datasets to Use First

1. MESA Sleep (NSRR): wrist actigraphy plus rich sleep phenotypes
- Fit: strong for sleep structure and night-level feature pretraining.
- Access: controlled research access.
- Source: https://sleepdata.org/datasets/mesa/pages/actigraphy-introduction.md

2. SHHS (NSRR): large cohort sleep study with PSG and cardiovascular outcomes
- Fit: large-scale sleep physiology pretraining and domain robustness.
- Access: controlled research access.
- Source: https://sleepdata.org/datasets/shhs

3. Sleep-EDF Expanded (PhysioNet): open sleep recordings and hypnograms
- Fit: open benchmark for sleep stage representation learning.
- Access: open.
- Source: https://physionet.org/content/sleep-edfx/

4. PPG-DaLiA (UCI): wrist PPG + accelerometer under daily activities
- Fit: robust HR/PPG feature extraction under motion artifacts.
- Access: open (CC BY 4.0).
- Source: https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA

5. WESAD (UCI): wrist/chest physiological signals across affective states
- Fit: stress/autonomic variability representations; auxiliary robustness task.
- Access: open (CC BY 4.0).
- Source: https://archive-beta.ics.uci.edu/dataset/465/wesad%2Bwearable%2Bstress%2Band%2Baffect%2Bdetection

## Mobile-First Training Strategy

### Stage A: Public Pretraining (offline)
- Train representation models on public tasks:
  - sleep stage proxy
  - HR trend estimation
  - stress state classification
- Keep architecture small from day one (tiny MLP/1D-CNN).

### Stage B: Feature Transfer
- Export compact embeddings or summary features.
- Concatenate with your current handcrafted night features.

### Stage C: Task Fine-Tuning (your fatigue labels)
- Train on your time-ordered split.
- Use time-series cross-validation for model selection.
- Optimize weighted F1 + calibration.

### Stage D: Distillation to Edge Model
- Teacher: stronger offline model (can be larger).
- Student: mobile model under constraints:
  - watch target <= 500 KB
  - phone target <= 1 MB
  - single prediction latency <= 250 ms watch, <= 100 ms phone
- Export to:
  - linear contract JSON (already supported)
  - CoreML artifact for iOS runtime

## Research-Quality Evaluation Protocol

1. Two evaluation tracks
- Full-history track: reflects deployment reality.
- Stable-sensor track: isolates upper-bound model capability.

2. Required ablations
- no public pretraining vs public pretraining
- handcrafted-only vs handcrafted + transferred features
- no distillation vs distilled edge model

3. Robustness tests
- synthetic sensor dropout (10/30/50%)
- sensor noise injection
- missingness pattern shift across time

4. Statistical reporting
- report mean/std via walk-forward CV
- add bootstrap confidence intervals for final metrics

## Practical Constraints and Risks

1. Label mismatch risk
- Public datasets usually do not have "next-day fatigue" labels.
- Use them for representation learning, not direct final supervision.

2. Domain shift
- Sensor hardware and sampling rates differ from Apple Watch/HealthKit.
- Add normalization and domain-adaptation checks.

3. Licensing and data-use terms
- Respect each dataset's terms, citation, and redistribution limits.

## Implementation Backlog (Concrete)

1. Add script: `scripts/step6_public_pretrain.py`
- public dataset loaders and auxiliary tasks

2. Add script: `scripts/step7_transfer_finetune.py`
- merges transferred embeddings with existing nightly features

3. Extend script: `scripts/step5_train_mobile_models.py`
- add time-series CV report
- add teacher-student distillation path

4. Add report outputs
- `reports/public_transfer_ablation.csv`
- `reports/robustness_metrics.json`
- `reports/mobile_tradeoff_table.csv`

## Success Criteria

- Increase weighted F1 by >= 0.05 over current baseline on held-out time split.
- Preserve mobile constraints (size + latency budgets).
- Demonstrate robustness under at least 30% synthetic missingness.
