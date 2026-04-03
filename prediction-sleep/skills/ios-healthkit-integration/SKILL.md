---
name: ios-healthkit-integration
description: "Use this skill when implementing or reviewing iOS SwiftUI + HealthKit data access for on-device mobile-computing demos."
---

# iOS HealthKit Integration Skill

## When to use

Use this skill when:
- Adding HealthKit authorization to a SwiftUI app
- Fetching Apple Health data in iOS
- Validating edge/on-device flow for demo readiness

## Primary reference

- https://www.createwithswift.com/reading-data-from-healthkit-in-a-swiftui-app/

## Core workflow

1. Enable HealthKit capability in Xcode target.
2. Add privacy usage strings in app Info settings.
3. Create a HealthKit manager for authorization and queries.
4. Fetch health samples asynchronously.
5. Convert raw samples into model features.
6. Run prediction locally on device (no network dependency).
7. Add simulator-safe demo fallback for UI testing.

## Project-specific mapping

- Authorization/query layer: `ios_healthkit_demo/HealthKitManager.swift`
- Local model contract scoring: `ios_healthkit_demo/FatigueModel.swift`
- Demo UI/state flow: `ios_healthkit_demo/ContentView.swift`

## Validation checklist

- [ ] Health access request appears correctly
- [ ] Real device fetch works with Apple Watch synced data
- [ ] Simulator path works via `Load Demo Data`
- [ ] Prediction executes locally and shows latency

