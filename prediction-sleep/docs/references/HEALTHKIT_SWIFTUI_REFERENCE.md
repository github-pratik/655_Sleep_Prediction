# HealthKit SwiftUI Reference

## Source

- Title: `Reading data from HealthKit in a SwiftUI app`
- Author: Matteo Altobello
- Published: April 22, 2025
- URL: https://www.createwithswift.com/reading-data-from-healthkit-in-a-swiftui-app/

## Why This Is Useful For Our Project

This article is directly relevant to our iPhone edge-demo path because it covers:
- HealthKit capability and privacy usage setup
- Requesting authorization from SwiftUI
- Reading HealthKit samples asynchronously
- ViewModel-driven UI updates after health queries

Our current app (`ios_healthkit_demo/`) uses the same core flow:
1. Ask HealthKit authorization
2. Query health records
3. Compute local features
4. Predict locally with no server call

## Key Patterns We Should Keep

1. Centralized HealthKit manager object for authorization + query logic.
2. Explicit list of read types for least-privilege access.
3. Async query handling to avoid blocking UI.
4. Clear user-facing error and authorization state handling.

## Adaptation Notes For Our Use Case

The tutorial demonstrates step count, active energy, and heart rate samples.
For our fatigue model, we extend that pattern to:
- Sleep analysis (night segmentation)
- HR / HRV / respiratory rate / SpO2 summary statistics
- Local feature imputation before inference

## Integration Checklist

- [ ] HealthKit capability enabled in target
- [ ] `NSHealthShareUsageDescription` present
- [ ] Query failures surfaced in UI state
- [ ] Simulator fallback path (`Load Demo Data`) available
- [ ] Physical-device end-to-end test completed

