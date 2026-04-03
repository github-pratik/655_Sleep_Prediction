# iOS HealthKit Edge Demo

This demo is a native SwiftUI app path for your CS655 mobile-computing framing:
- Reads Apple Health data directly on iPhone (including Apple Watch-synced data)
- Computes features on-device
- Runs fatigue prediction on-device (no cloud inference)

## Included Files

- `SleepFatigueEdgeApp.swift`
- `ContentView.swift`
- `HealthKitManager.swift`
- `FatigueModel.swift`
- `mobile_linear_contract.json`

## Xcode Setup (Physical iPhone Required)

1. Create a new iOS App project in Xcode (SwiftUI).
2. Drag all files from this folder into your Xcode project.
3. Ensure `mobile_linear_contract.json` is in **Copy Bundle Resources**.
4. In target settings:
   - **Signing & Capabilities** -> add **HealthKit**
5. In `Info.plist`, add:
   - `NSHealthShareUsageDescription` = "This app reads sleep and vitals to compute fatigue prediction on-device."
6. Build and run on a real iPhone (HealthKit data is not available in simulator).

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

