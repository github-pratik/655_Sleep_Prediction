# Mobile Benchmark Template

Use this template to capture device-side measurements for the fatigue model.
Keep entries tied to a specific model artifact and OS/build version.

## Run Metadata

- Date:
- Device:
- OS version:
- App/runtime:
- Model artifact:
- Feature contract:
- Notes:

## Latency

Record the end-to-end prediction path, not just the raw estimator.

- Warm-up runs:
- Sample count:
- Median latency (ms):
- P95 latency (ms):
- Max latency (ms):
- Background load:

## Memory

Capture memory during model load and inference.

- Baseline app memory:
- Peak memory during model load:
- Peak memory during prediction:
- Memory after idle:
- Notes on spikes:

## Battery

Capture battery impact on-device under repeated predictions.

- Starting battery:
- Ending battery:
- Session duration:
- Prediction count:
- Approximate drain per hour:
- Notes on thermal state:

## Accuracy Check

Use a fixed test set or fixed demo inputs to verify model consistency.

- Test input source:
- Expected label / output:
- Observed label / output:
- Parity check passed:
- Notes:

## Failure Modes

Record missing-sensor behavior and fallbacks.

- Missing sleep fields:
- Missing HR/HRV fields:
- Missing respiratory fields:
- Missing SpO2 fields:
- Fallback behavior:
- Crash observed:

## Summary

- Overall readiness:
- Main blocker:
- Next action:
