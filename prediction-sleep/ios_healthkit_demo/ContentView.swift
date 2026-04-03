import SwiftUI

struct ContentView: View {
    @StateObject private var healthKit = HealthKitManager()
    @State private var features: [String: Double] = [:]
    @State private var prediction: FatiguePrediction?
    @State private var uiMessage = "Tap \"Request Health Access\" first."
    @State private var loading = false

    private let model = try? FatigueModel()

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

    private func format(_ value: Double?) -> String {
        guard let value else { return "-" }
        return String(format: "%.3f", value)
    }
}
