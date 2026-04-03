# iOS HealthKit Edge Demo

This demo is a native SwiftUI app path for CS655 mobile-computing framing:
- Reads Apple Health data directly on iPhone (including Apple Watch-synced data)
- Computes features on-device
- Runs fatigue prediction on-device (no cloud inference)

## Included Files

- `SleepFatigueEdgeApp.swift`
- `ContentView.swift`
- `HealthKitManager.swift`
- `FatigueModel.swift`
- `mobile_linear_contract.json`
- `project.yml` (XcodeGen project spec)
- `SleepFatigueEdgeDemo.entitlements`
- `generate_project.sh`

## Fast Setup (Recommended)

From this folder:

```bash
chmod +x generate_project.sh
./generate_project.sh
```

This generates and opens:
- `SleepFatigueEdgeDemo.xcodeproj`

Then in Xcode:
1. Select your Apple Developer Team under **Signing & Capabilities**.
2. Plug in your iPhone and run the app.

Notes:
- The project already includes HealthKit entitlement and usage descriptions.
- HealthKit data is unavailable in iOS simulator, so use a physical iPhone.

## File to Open in Xcode

Open:
- `SleepFatigueEdgeDemo.xcodeproj`

from folder:
- `prediction-sleep/ios_healthkit_demo/`

## Simulator Testing Mode

You can still test app flow in Simulator:
1. Run the app in iOS Simulator.
2. Tap **Load Demo Data (Simulator Safe)**.
3. Tap **Run On-Device Prediction**.

This validates:
- UI flow
- Local preprocessing/inference path
- Prediction rendering

Use physical iPhone for real HealthKit fetch.

## What Data Is Read

- Sleep analysis (`HKCategoryTypeIdentifierSleepAnalysis`)
- Heart rate
- HRV (SDNN)
- Respiratory rate
- Oxygen saturation

The app fetches the latest night block (using a night-key shift), computes summary features, and predicts locally.

## Notes

- Apple Watch data appears in HealthKit after sync to iPhone.
- If no sleep data exists for the recent window, fetch will fail until data is present.
- For production, add stronger error handling and background refresh scheduling.
