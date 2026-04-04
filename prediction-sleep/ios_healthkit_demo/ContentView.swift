import SwiftUI

struct ContentView: View {
    @StateObject private var healthKit = HealthKitManager()
    @State private var features: [String: Double] = [:]
    @State private var prediction: FatiguePrediction?
    @State private var uiMessage = "Tap \"Request Health Access\" first."
    @State private var loading = false
    @State private var lastSyncAt: Date?
    @State private var lastInferenceMs: Double?

    // Cold-start / readiness
    @State private var readinessState: ReadinessState = .noData
    @State private var readinessChecked = false

    // Multi-night history
    @State private var nightHistory: [NightSleepSummary] = []
    @State private var historyPredictions: [FatiguePrediction] = []
    @State private var showHistory = false

    private let model = try? FatigueModel()

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

                    // Readiness / cold-start banner
                    if readinessChecked && !showHistory {
                        readinessBanner
                    }

                    if !features.isEmpty && !showHistory {
                        nightSummaryCard
                    }

                    if let prediction, !showHistory {
                        predictionCard(prediction)
                    }

                    if !features.isEmpty && !showHistory {
                        robustnessCard
                        edgeRuntimeCard
                    }

                    // 7-night trend view
                    if showHistory && !nightHistory.isEmpty {
                        historyTrendView
                    }

                    if let model, features.isEmpty && !showHistory {
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

    // MARK: - Readiness Banner

    private var readinessBanner: some View {
        let state = readinessState
        let icon: String
        let color: Color
        let message: String

        switch state {
        case .noData:
            icon = "📡"
            color = .orange
            message = "No sleep data found in HealthKit. Wear your Apple Watch to sleep tonight and check back tomorrow."
        case .collectingBaseline:
            icon = "⏳"
            color = .orange
            message = "Collecting your baseline. You have \(healthKit.dataAvailability.availableHistoryDays) day(s) of history. Personalized predictions available in \(max(0, 7 - healthKit.dataAvailability.availableHistoryDays)) days."
        case .partialPersonalization:
            icon = "📊"
            color = .blue
            message = "Predictions active. You have \(healthKit.dataAvailability.availableHistoryDays) day(s) of history. Accuracy will continue improving over the next 30 days."
        case .fullPersonalization:
            icon = "✅"
            color = .green
            message = "Fully personalized predictions enabled. \(healthKit.dataAvailability.availableHistoryDays) days of your health data establish your unique baseline."
        }

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(icon) Data Readiness")
                    .font(.headline)
                Spacer()
                Text(state.label)
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(color.opacity(0.15))
                    .foregroundStyle(color)
                    .clipShape(Capsule())
            }
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            // Data source breakdown
            VStack(alignment: .leading, spacing: 4) {
                let avail = healthKit.dataAvailability
                readinessRow("Sleep", avail.hasSleepData)
                readinessRow("Heart Rate", avail.hasHR)
                readinessRow("HRV (SDNN)", avail.hasHRV)
                readinessRow("Respiratory Rate", avail.hasResp)
                readinessRow("SpO₂", avail.hasSpO2)
                readinessRow("Steps", avail.hasSteps)
                readinessRow("Active Energy", avail.hasActiveEnergy)
                readinessRow("Exercise Time", avail.hasWorkout)
            }
            .font(.caption)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func readinessRow(_ name: String, _ available: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: available ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(available ? .green : .gray)
            Text(name)
            Spacer()
            Text(available ? "Available" : "Not detected")
                .foregroundStyle(available ? .green : .gray)
                .font(.caption2)
        }
    }

    // MARK: - History Trend View

    private var historyTrendView: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("📈 Last 7 Nights Trend")
                    .font(.headline)
                Spacer()
                Button("Back") {
                    showHistory = false
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Trend cards
            ForEach(Array(nightHistory.enumerated()), id: \.offset) { index, night in
                let pred = index < historyPredictions.count ? historyPredictions[index] : nil
                NightTrendCard(night: night, prediction: pred, dateFormatter: dateFormatter)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var dateFormatter: DateFormatter {
        let f = DateFormatter()
        f.dateFormat = "EEE M/d"
        return f
    }

    // MARK: - Existing Views

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
            Text("Data origin: \(healthKit.lastDataOrigin)")
                .foregroundStyle(.secondary)
                .font(.footnote)
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

            Button("Fetch + Predict") {
                fetchLatestNightAndPredict()
            }
            .buttonStyle(.borderedProminent)
            .disabled(!healthKit.authorizationGranted || model == nil || loading)

            HStack(spacing: 10) {
                Button("Load Demo Data") {
                    loadDemoFeatures()
                }
                .buttonStyle(.bordered)
                .disabled(model == nil || loading)

                Button("Run Prediction") {
                    runPrediction()
                }
                .buttonStyle(.borderedProminent)
                .disabled(features.isEmpty || model == nil || loading)
            }

            // View history — now enabled when HealthKit connected
            Button("View Last 7 Nights") {
                fetchHistory()
            }
            .buttonStyle(.bordered)
            .disabled(!healthKit.authorizationGranted || loading)
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

            // Activity features
            if let steps = features["steps"], steps > 0 {
                summaryRow("Steps", "\(Int(steps).formatted())")
            }
            if let energy = features["active_energy"], energy > 0 {
                summaryRow("Active Energy", "\(Int(energy)) kcal")
            }
            if let effort = features["avg_physical_effort"], effort > 0 {
                summaryRow("Exercise Time", "\(Int(effort)) min")
            }

            // Rolling features (if present)
            if let rolling3d = features["total_sleep_minutes_rolling_3d_mean"], rolling3d.isFinite {
                summaryRow("3d Avg Sleep", "\(Int(rolling3d)) min")
            }
            if let rolling7d = features["total_sleep_minutes_rolling_7d_mean"], rolling7d.isFinite {
                summaryRow("7d Avg Sleep", "\(Int(rolling7d)) min")
            }
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
            if let threshold = model?.contract.decisionThreshold {
                Text("Decision threshold: \(format(threshold))")
                    .foregroundStyle(.secondary)
                    .font(.footnote)
            }
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
            let sizeKB = contractSizeKB
            Text("Model size: \(format(sizeKB)) KB")
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

    // MARK: - Helpers

    private var referenceP95Ms: Double { 1.0 }

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
        guard let m else { return 0 }
        return m.contract.featureOrder.reduce(0) { partial, feature in
            guard let value = features[feature], value.isFinite else { return partial + 1 }
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
              let size = attrs[.size] as? NSNumber else { return 0 }
        return size.doubleValue / 1024.0
    }

    // MARK: - Actions

    private func requestAccess() {
        loading = true
        healthKit.requestAuthorization { result in
            DispatchQueue.main.async {
                loading = false
                switch result {
                case .success:
                    uiMessage = "Access granted. Checking data availability..."
                    checkReadiness()
                case .failure(let error):
                    uiMessage = "Authorization failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func checkReadiness() {
        healthKit.checkDataAvailability { availability in
            DispatchQueue.main.async {
                readinessState = availability.readinessState
                readinessChecked = true
                if availability.readinessState == .noData {
                    uiMessage = "Health access granted but no sleep data found yet. Wear your watch to sleep tonight!"
                } else {
                    uiMessage = "Access granted. \(availability.availableHistoryDays) day(s) of history available."
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

    private func fetchLatestNightAndPredict() {
        loading = true
        healthKit.fetchLatestNightFeatures { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let fetched):
                    features = fetched
                    prediction = nil
                    lastSyncAt = Date()
                    runPrediction()
                    loading = false
                    uiMessage = "Fetched Apple Health data and predicted on-device."
                case .failure(let error):
                    loading = false
                    uiMessage = "Fetch + predict failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func fetchHistory() {
        loading = true
        showHistory = true
        historyPredictions = []
        healthKit.fetchMultipleNights(count: 7) { result in
            DispatchQueue.main.async {
                loading = false
                switch result {
                case .success(let nights):
                    nightHistory = nights
                    // Run prediction for each night
                    guard let model else { return }
                    for night in nights {
                        var filledFeatures = model.completeFeatures(night.features)
                        // Fill rolling/lag from imputer (not available from single-night fetch)
                        let pred = model.predict(features: filledFeatures)
                        historyPredictions.append(pred)
                    }
                case .failure(let error):
                    showHistory = false
                    uiMessage = "History fetch failed: \(error.localizedDescription)"
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

        // Core sleep
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

        // HR
        demo["hr_mean"] = 64.0
        demo["hr_min"] = 52.0
        demo["hr_max"] = 78.0
        demo["hr_median"] = 63.0
        demo["hr_std"] = 4.1

        // HRV
        demo["hrv_mean"] = 58.0
        demo["hrv_min"] = 42.0
        demo["hrv_max"] = 88.0
        demo["hrv_median"] = 55.0
        demo["hrv_std"] = 10.5

        // Resp
        demo["resp_mean"] = 15.2
        demo["resp_min"] = 13.4
        demo["resp_max"] = 18.1
        demo["resp_median"] = 15.1
        demo["resp_std"] = 0.9

        // SpO2
        demo["spo2_mean"] = 0.982
        demo["spo2_min"] = 0.967
        demo["spo2_max"] = 0.992
        demo["spo2_median"] = 0.982
        demo["spo2_std"] = 0.004

        // Activity
        demo["steps"] = 8500.0
        demo["active_energy"] = 350.0
        demo["avg_physical_effort"] = 28.0

        // Rolling 3d
        demo["total_sleep_minutes_rolling_3d_mean"] = 400.0
        demo["total_sleep_minutes_rolling_3d_std"] = 30.0
        demo["sleep_efficiency_rolling_3d_mean"] = 0.88
        demo["sleep_efficiency_rolling_3d_std"] = 0.05
        demo["deep_minutes_rolling_3d_mean"] = 70.0
        demo["deep_minutes_rolling_3d_std"] = 10.0
        demo["rem_minutes_rolling_3d_mean"] = 88.0
        demo["rem_minutes_rolling_3d_std"] = 12.0
        demo["steps_rolling_3d_mean"] = 8200.0
        demo["steps_rolling_3d_std"] = 1500.0
        demo["active_energy_rolling_3d_mean"] = 330.0
        demo["active_energy_rolling_3d_std"] = 50.0

        // Rolling 7d
        demo["total_sleep_minutes_rolling_7d_mean"] = 390.0
        demo["total_sleep_minutes_rolling_7d_std"] = 45.0
        demo["sleep_efficiency_rolling_7d_mean"] = 0.86
        demo["sleep_efficiency_rolling_7d_std"] = 0.07
        demo["deep_minutes_rolling_7d_mean"] = 65.0
        demo["deep_minutes_rolling_7d_std"] = 15.0
        demo["rem_minutes_rolling_7d_mean"] = 85.0
        demo["rem_minutes_rolling_7d_std"] = 18.0
        demo["steps_rolling_7d_mean"] = 7800.0
        demo["steps_rolling_7d_std"] = 2000.0
        demo["active_energy_rolling_7d_mean"] = 310.0
        demo["active_energy_rolling_7d_std"] = 70.0

        // Lag
        demo["total_sleep_minutes_lag1"] = 395.0
        demo["sleep_efficiency_lag1"] = 0.87
        demo["steps_lag1"] = 7500.0

        // Day of week
        demo["day_of_week"] = 3.0

        features = demo
        prediction = nil
        lastSyncAt = Date()
        uiMessage = "Loaded demo features (61 features). Tap \"Run Prediction\"."
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

// MARK: - NightTrendCard

struct NightTrendCard: View {
    let night: NightSleepSummary
    let prediction: FatiguePrediction?
    let dateFormatter: DateFormatter

    var body: some View {
        let riskIsHigh = prediction?.label == 1
        let confidence = prediction.map { $0.label == 1 ? $0.probability1 : $0.probability0 }

        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(dateFormatter.string(from: night.nightDate))
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
                if let prediction {
                    Text(riskIsHigh ? "HIGH" : "LOW")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(riskIsHigh ? Color.red.opacity(0.15) : Color.green.opacity(0.15))
                        .foregroundStyle(riskIsHigh ? .red : .green)
                        .clipShape(Capsule())
                    Text("\(Int((confidence ?? 0) * 100))%")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text("No prediction")
                        .font(.caption2)
                        .foregroundStyle(.gray)
                }
            }

            HStack(spacing: 12) {
                miniMetric("Sleep", "\(Int(night.features["total_sleep_minutes"] ?? 0))m")
                miniMetric("Efficiency", "\(Int((night.features["sleep_efficiency"] ?? 0) * 100))%")
                miniMetric("HR", "\(Int(night.features["hr_mean"] ?? 0))")
                miniMetric("HRV", "\(Int(night.features["hrv_mean"] ?? 0))")
                if let steps = night.features["steps"], steps > 0 {
                    miniMetric("Steps", "\(Int(steps).formatted())")
                }
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Color(uiColor: .tertiarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func miniMetric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit())
                .fontWeight(.medium)
        }
    }
}

// MARK: - ReadinessState extension

extension ReadinessState {
    var label: String {
        switch self {
        case .noData: return "No Data"
        case .collectingBaseline: return "Collecting Baseline"
        case .partialPersonalization: return "Partial"
        case .fullPersonalization: return "Fully Personalized"
        }
    }
}
