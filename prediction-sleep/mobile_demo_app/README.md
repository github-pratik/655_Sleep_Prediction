# Mobile Demo App (Edge Inference)

This is a lightweight Expo React Native demo that runs fatigue prediction fully on-device using the exported linear model contract.

## Why this is Mobile-Computing Aligned

- No server dependency at inference time
- Feature preprocessing + model scoring run locally on phone
- Small model contract artifact suitable for edge deployment
- Easy to profile latency and battery in realistic mobile runs

## Files

- `App.js`: Demo UI for entering feature values and running prediction
- `src/inference.js`: Pure JS inference implementation
- `src/contract.json`: Exported contract from `artifacts/mobile_linear_contract.json`

## Run

```bash
cd mobile_demo_app
npm install
npx expo start
```

Then open in Expo Go on your phone.

## Refresh Contract After Retraining

From project root:

```bash
.venv/bin/python scripts/step5_train_mobile_models.py --skip-plots
cp artifacts/mobile_linear_contract.json mobile_demo_app/src/contract.json
```

