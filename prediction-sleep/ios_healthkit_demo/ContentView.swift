import SwiftUI

struct ContentView: View {
    @StateObject private var healthKit = HealthKitManager()
    @State private var features: [String: Double] = [:]
    @State private var prediction: FatiguePrediction?
    @State private var uiMessage = "Tap \"Request Health Access\" first."
    @State private var loading = false

    private let model = try? FatigueModel()
    private let isSimulator: Bool = {
#if targetEnvironment(simulator)
        true
#else
        false
#endif
    }()

    private let displayFeatureOrder: [String] = [
        "total_sleep_minutes",
        "sleep_efficiency",
        "rem_pct",
        "deep_pct",
        "core_pct",
        "hr_mean",
        "hr_std",
        "hrv_mean",
        "resp_mean",
        "spo2_mean"
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Sleep Fatigue Edge Demo")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Reads Apple Health data from iPhone/Apple Watch and predicts on-device.")
                        .foregroundStyle(.secondary)

                    statusCard

                    HStack(spacing: 12) {
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

                    if !features.isEmpty {
                        featureCard
                    }

                    if let prediction {
                        predictionCard(prediction)
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

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Status")
                .font(.headline)
            Text("HealthKit: \(healthKit.statusMessage)")
            Text("Environment: \(isSimulator ? "Simulator (Demo Mode recommended)" : "Device")")
                .foregroundStyle(.secondary)
            Text(uiMessage)
                .foregroundStyle(.secondary)
            if loading {
                ProgressView()
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var featureCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Latest Night Features")
                .font(.headline)
            ForEach(displayFeatureOrder, id: \.self) { key in
                HStack {
                    Text(key)
                        .font(.caption)
                    Spacer()
                    Text(format(features[key]))
                        .font(.caption.monospacedDigit())
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func predictionCard(_ p: FatiguePrediction) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Prediction")
                .font(.headline)
            Text("Label: \(p.label)")
            Text(String(format: "P(0): %.3f", p.probability0))
            Text(String(format: "P(1): %.3f", p.probability1))
            Text(String(format: "Logit: %.3f", p.logit))
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
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
        prediction = model.predict(features: features)
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
        uiMessage = "Loaded demo features locally. Now tap \"Run On-Device Prediction\"."
    }

    private func format(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.3f", value)
    }
}
