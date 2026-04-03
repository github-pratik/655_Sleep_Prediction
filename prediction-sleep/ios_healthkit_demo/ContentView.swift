import SwiftUI

struct ContentView: View {
    @StateObject private var healthKit = HealthKitManager()
    @State private var features: [String: Double] = [:]
    @State private var prediction: FatiguePrediction?
    @State private var uiMessage = "Tap \"Request Health Access\" first."
    @State private var loading = false
    @State private var lastSyncAt: Date?
    @State private var lastInferenceMs: Double?

    private let model = try? FatigueModel()
    private let referenceP95Ms = 1.0

    private let isSimulator: Bool = {
#if targetEnvironment(simulator)
        true
#else
        false
#endif
    }()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Sleep Fatigue Edge Demo")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Apple Watch + Health data analyzed locally on your iPhone.")
                        .foregroundStyle(.secondary)

                    if isSimulator {
                        simulatorBanner
                    }

                    statusCard
                    actionButtons

                    if !features.isEmpty {
                        nightSummaryCard
                    }

                    if let prediction {
                        predictionCard(prediction)
                    }

                    if !features.isEmpty {
                        robustnessCard
                        edgeRuntimeCard
                    }

                    if let model, features.isEmpty {
                        Text("Model loaded with \(model.contract.featureOrder.count) features.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
            }
            .navigationTitle("Mobile Computing Demo")
        }
    }

    private var simulatorBanner: some View {
        Text("Simulator mode: HealthKit live data is unavailable. Use \"Load Demo Data\".")
            .font(.footnote)
            .foregroundStyle(Color.orange)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.orange.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Status")
                .font(.headline)
            Text("HealthKit: \(healthKit.authorizationGranted ? "Connected" : "Not Connected")")
            Text("Runtime: On-device (No Cloud)")
            Text("Source: Apple Health (Watch synced)")
            Text("Last sync: \(formatSyncTime(lastSyncAt))")
                .foregroundStyle(.secondary)
            Text(uiMessage)
                .foregroundStyle(.secondary)
                .font(.footnote)
            if loading {
                ProgressView()
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var actionButtons: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button("Request Health Access") {
                    requestAccess()
                }
                .buttonStyle(.borderedProminent)

                Button("Fetch Latest Night") {
                    fetchLatestNight()
                }
                .buttonStyle(.bordered)
                .disabled(!healthKit.authorizationGranted || loading)
            }

            HStack(spacing: 10) {
                Button("Load Demo Data (Simulator Safe)") {
                    loadDemoFeatures()
                }
                .buttonStyle(.bordered)
                .disabled(model == nil || loading)

                Button("Run On-Device Prediction") {
                    runPrediction()
                }
                .buttonStyle(.borderedProminent)
                .disabled(features.isEmpty || model == nil || loading)
            }

            Button("View Last 7 Nights") {}
                .buttonStyle(.bordered)
                .disabled(true)
        }
    }

    private var nightSummaryCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Latest Night Summary")
                .font(.headline)

            summaryRow("Total Sleep", "\(format(features["total_sleep_minutes"])) min")
            summaryRow("Sleep Efficiency", "\(formatPct(features["sleep_efficiency"]))%")
            summaryRow(
                "REM / Deep / Core",
                "\(formatPct(features["rem_pct"]))% / \(formatPct(features["deep_pct"]))% / \(formatPct(features["core_pct"]))%"
            )
            summaryRow("Avg HR", "\(format(features["hr_mean"])) bpm")
            summaryRow("Avg HRV (SDNN)", "\(format(features["hrv_mean"])) ms")
            summaryRow("Avg Resp", "\(format(features["resp_mean"])) /min")
            summaryRow("Avg SpO₂", "\(formatPct(features["spo2_mean"]))%")
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func predictionCard(_ p: FatiguePrediction) -> some View {
        let riskIsHigh = p.label == 1
        let confidence = riskIsHigh ? p.probability1 : p.probability0

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Next-Day Fatigue Risk")
                    .font(.headline)
                Spacer()
                Text(riskIsHigh ? "HIGH" : "LOW")
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(riskIsHigh ? Color.red.opacity(0.18) : Color.green.opacity(0.2))
                    .foregroundStyle(riskIsHigh ? Color.red : Color.green)
                    .clipShape(Capsule())
            }

            Text("Confidence: \(Int((confidence * 100).rounded()))%")
            Text("Computed locally in \(format(lastInferenceMs)) ms")
            Text("No server request was used.")
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var robustnessCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Data Robustness")
                .font(.headline)
            Text("Missing features: \(missingFeatureCount)/\(totalFeatureCount)")
            Text("Fallback: Local imputation active")
            Text("Data quality: \(dataQualityLabel)")
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var edgeRuntimeCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Edge Runtime")
                .font(.headline)
            Text("Model: Linear Contract (Logistic)")
            Text("Model size: \(format(contractSizeKB)) KB")
            Text("Inference p95: \(format(referenceP95Ms)) ms")
            if let lastInferenceMs {
                Text("Last inference: \(format(lastInferenceMs)) ms")
            }
            Text("Network required: No")
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func summaryRow(_ key: String, _ value: String) -> some View {
        HStack {
            Text(key)
                .font(.caption)
            Spacer()
            Text(value)
                .font(.caption.monospacedDigit())
        }
    }

    private var totalFeatureCount: Int {
        model?.contract.featureOrder.count ?? 0
    }

    private var missingFeatureCount: Int {
        guard let model else { return 0 }
        return model.contract.featureOrder.reduce(0) { partial, feature in
            guard let value = features[feature], value.isFinite else {
                return partial + 1
            }
            return partial
        }
    }

    private var dataQualityLabel: String {
        if missingFeatureCount <= 2 { return "Good" }
        if missingFeatureCount <= 8 { return "Fair" }
        return "Poor"
    }

    private var contractSizeKB: Double {
        guard let url = Bundle.main.url(forResource: "mobile_linear_contract", withExtension: "json"),
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else {
            return 0
        }
        return size.doubleValue / 1024.0
    }

    private func requestAccess() {
        loading = true
        healthKit.requestAuthorization { result in
            DispatchQueue.main.async {
                loading = false
                switch result {
                case .success:
                    uiMessage = "Access granted. You can fetch the latest night."
                case .failure(let error):
                    uiMessage = "Authorization failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func fetchLatestNight() {
        loading = true
        healthKit.fetchLatestNightFeatures { result in
            DispatchQueue.main.async {
                loading = false
                switch result {
                case .success(let fetched):
                    features = fetched
                    prediction = nil
                    lastSyncAt = Date()
                    uiMessage = "Fetched latest night from Apple Health."
                case .failure(let error):
                    uiMessage = "Data fetch failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func runPrediction() {
        guard let model else {
            uiMessage = "Model contract not loaded."
            return
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        prediction = model.predict(features: features)
        let t1 = CFAbsoluteTimeGetCurrent()
        lastInferenceMs = (t1 - t0) * 1000.0
        uiMessage = "Prediction ran on-device (no server)."
    }

    private func loadDemoFeatures() {
        guard let model else {
            uiMessage = "Model contract not loaded."
            return
        }

        var demo = model.contract.imputerMedian
        demo["total_sleep_minutes"] = 415.0
        demo["asleep_minutes"] = 415.0
        demo["in_bed_minutes"] = 450.0
        demo["sleep_efficiency"] = 415.0 / 450.0
        demo["rem_minutes"] = 92.0
        demo["deep_minutes"] = 74.0
        demo["core_minutes"] = 249.0
        demo["rem_pct"] = 92.0 / 415.0
        demo["deep_pct"] = 74.0 / 415.0
        demo["core_pct"] = 249.0 / 415.0

        demo["hr_mean"] = 64.0
        demo["hr_min"] = 52.0
        demo["hr_max"] = 78.0
        demo["hr_median"] = 63.0
        demo["hr_std"] = 4.1

        demo["hrv_mean"] = 58.0
        demo["hrv_min"] = 42.0
        demo["hrv_max"] = 88.0
        demo["hrv_median"] = 55.0
        demo["hrv_std"] = 10.5

        demo["resp_mean"] = 15.2
        demo["resp_min"] = 13.4
        demo["resp_max"] = 18.1
        demo["resp_median"] = 15.1
        demo["resp_std"] = 0.9

        demo["spo2_mean"] = 0.982
        demo["spo2_min"] = 0.967
        demo["spo2_max"] = 0.992
        demo["spo2_median"] = 0.982
        demo["spo2_std"] = 0.004

        features = demo
        prediction = nil
        lastSyncAt = Date()
        uiMessage = "Loaded demo features locally. Now tap \"Run On-Device Prediction\"."
    }

    private func format(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.2f", value)
    }

    private func formatPct(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.1f", value * 100)
    }

    private func formatSyncTime(_ value: Date?) -> String {
        guard let value else { return "Not synced yet" }
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .short
        return formatter.string(from: value)
    }
}

