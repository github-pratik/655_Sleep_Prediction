import SwiftUI

// UI built with SwiftUI patterns from Apple’s framework reference:
// https://developer.apple.com/documentation/swiftui

struct ContentView: View {
    @StateObject private var healthKit = HealthKitManager()
    @State private var features: [String: Double] = [:]
    @State private var prediction: FatiguePrediction?
    @State private var uiMessage = "Use the Status tab to connect Health or load demo data."
    @State private var loading = false
    @State private var lastSyncAt: Date?
    @State private var lastInferenceMs: Double?

    @State private var nightHistory: [NightSleepSummary] = []
    @State private var historyPredictions: [FatiguePrediction] = []

    @State private var model: FatigueModel?
    @State private var modelLoadError: String?
    @State private var modelLoadAttempted = false

    @State private var selectedTab: StitchMainTab = .status
    @State private var showPermissionSheet = false

    private let isSimulator: Bool = {
#if targetEnvironment(simulator)
        true
#else
        false
#endif
    }()

    var body: some View {
        VStack(spacing: 0) {
            StitchTopBar {
                showPermissionSheet = true
            }

            Group {
                switch selectedTab {
                case .status:
                    statusWithFeedbackTabScroll
                case .settings:
                    settingsTabScroll
                case .history:
                    historyTabScroll
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            StitchBottomBar(selection: $selectedTab)
        }
        .background(ClinicalTheme.canvasAdaptive)
        .sheet(isPresented: $showPermissionSheet) {
            PermissionOnboardingView(
                onRequestAccess: {
                    showPermissionSheet = false
                    requestAccess()
                },
                onSkip: {
                    showPermissionSheet = false
                }
            )
            .presentationDragIndicator(.visible)
        }
        .onAppear {
            healthKit.checkDataAvailability { _ in }

            guard !modelLoadAttempted else { return }
            modelLoadAttempted = true
            DispatchQueue.main.async {
                guard model == nil else { return }
                do {
                    model = try FatigueModel()
                    modelLoadError = nil
                } catch {
                    modelLoadError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                }
            }
        }
        .onChange(of: selectedTab) { _, newTab in
            if newTab == .history, nightHistory.isEmpty {
                fetchHistory()
            }
        }
    }

    // MARK: - Stitch tabs (`cold_start_permissions`, `readiness_dashboard`, `fatigue_prediction_results`, `7_night_history_trend`)

    private var emptyHistoryHint: String {
        if isSimulator {
            return "No HealthKit sleep nights in the Simulator. Tap Load Demo Data on Status (fills a demo 7-night trend here), or grant access and use a device with watch sleep data."
        }
        if !healthKit.authorizationGranted {
            return "Grant Health access on the Status tab, then open History again. Or use Open 7-night history from Settings after a successful fetch."
        }
        return "Loading or no nights in the selected window. Try Open 7-night history on Settings, or sync Apple Watch sleep to Health."
    }

    private var statusTabScroll: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                coldStartTitleBlock
                coldStartConnectionCard
                loadDemoOutlineButton
                researchPrivacyExplainer
                coldStartVisualHero
                statusFooterDetails
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)
            .padding(.bottom, 28)
        }
        .scrollIndicators(.visible)
        .scrollContentBackground(.hidden)
        .background(ClinicalTheme.canvasAdaptive)
    }

    private var coldStartTitleBlock: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sleep Fatigue Edge Demo")
                .font(.system(size: 34, weight: .heavy, design: .rounded))
                .foregroundStyle(ClinicalTheme.onSurface)
                .accessibilityAddTraits(.isHeader)
            HStack(alignment: .top, spacing: 0) {
                Rectangle()
                    .fill(ClinicalTheme.primary.opacity(0.2))
                    .frame(width: 2)
                    .padding(.trailing, 12)
                Text("Watch + Health analyzed locally")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.bottom, 4)
    }

    private var coldStartConnectionCard: some View {
        let connected = healthKit.authorizationGranted
        return ZStack(alignment: .topTrailing) {
            VStack(alignment: .leading, spacing: 20) {
                HStack(alignment: .center, spacing: 14) {
                    ZStack {
                        Circle()
                            .fill(connected ? ClinicalTheme.primary.opacity(0.14) : ClinicalTheme.errorContainer)
                            .frame(width: 48, height: 48)
                        Image(systemName: "heart.text.square.fill")
                            .font(.title2)
                            .foregroundStyle(connected ? ClinicalTheme.primary : ClinicalTheme.errorRed)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Connection Status")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                            .textCase(.uppercase)
                            .tracking(0.8)
                        Text(connected ? "HealthKit Connected" : "HealthKit Not Connected")
                            .font(.title3.weight(.bold))
                            .foregroundStyle(connected ? ClinicalTheme.primary : ClinicalTheme.errorRed)
                    }
                }
                HStack(spacing: 12) {
                    Image(systemName: "lock.shield.fill")
                        .foregroundStyle(ClinicalTheme.primary)
                    Text("On-device (No Cloud)")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(ClinicalTheme.onSurface)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(ClinicalTheme.surfaceContainerLow)
                }
                Button("Request Health Access") {
                    requestAccess()
                }
                .buttonStyle(ColdStartPrimaryCTAButtonStyle())
                .disabled(loading)
            }
            .padding(22)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .fill(ClinicalTheme.cardFill)
                    .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 3)
            }
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 72))
                .foregroundStyle(ClinicalTheme.primary.opacity(0.05))
                .offset(x: 8, y: -4)
                .accessibilityHidden(true)
        }
    }

    private var loadDemoOutlineButton: some View {
        Button {
            loadDemoFeatures()
            selectedTab = .settings
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "play.circle.fill")
                    .font(.title3)
                Text("Load Demo Data")
                    .font(.headline.weight(.bold))
                    .fontDesign(.rounded)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .foregroundStyle(ClinicalTheme.primary)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .strokeBorder(ClinicalTheme.primary, lineWidth: 2)
            }
        }
        .buttonStyle(.plain)
        .disabled(loading)
    }

    private var researchPrivacyExplainer: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "eye.slash.fill")
                .font(.title2)
                .foregroundStyle(ClinicalTheme.primary)
            VStack(alignment: .leading, spacing: 6) {
                Text("Research Privacy")
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(ClinicalTheme.onSecondaryContainer)
                Text("Your data stays on this device for research analysis. We use local differential privacy to ensure no personal identifiers ever leave your hardware.")
                    .font(.caption)
                    .foregroundStyle(ClinicalTheme.onSecondaryContainer.opacity(0.85))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                .fill(ClinicalTheme.secondaryContainer.opacity(0.35))
        }
        .overlay {
            RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                .strokeBorder(ClinicalTheme.secondaryContainer.opacity(0.35), lineWidth: 1)
        }
    }

    private var coldStartVisualHero: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(ClinicalTheme.surfaceContainerHigh)
                .aspectRatio(1, contentMode: .fit)
                .overlay {
                    LinearGradient(
                        colors: [
                            ClinicalTheme.primary.opacity(0.15),
                            ClinicalTheme.primaryContainer.opacity(0.35)
                        ],
                        startPoint: .topTrailing,
                        endPoint: .bottomLeading
                    )
                }
                .overlay {
                    Image(systemName: "applewatch")
                        .font(.system(size: 72))
                        .foregroundStyle(.white.opacity(0.5))
                }
            LinearGradient(
                colors: [ClinicalTheme.canvas, .clear],
                startPoint: .bottom,
                endPoint: .center
            )
            VStack(alignment: .leading, spacing: 8) {
                Text("Edge Computing")
                    .font(.system(size: 10, weight: .bold))
                    .tracking(2)
                    .textCase(.uppercase)
                    .foregroundStyle(ClinicalTheme.primary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(ClinicalTheme.primary.opacity(0.12))
                    .clipShape(Capsule())
                Text("Private. Local. Precise.")
                    .font(.title3.weight(.bold))
                    .fontDesign(.rounded)
                    .foregroundStyle(ClinicalTheme.onSurface)
            }
            .padding(22)
        }
    }

    private var statusFooterDetails: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Data origin: \(healthKit.lastDataOrigin)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text("Last sync: \(formatSyncTime(lastSyncAt))")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(uiMessage)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let modelLoadError {
                Text(modelLoadError)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
            if loading { ProgressView().padding(.top, 4) }
            if let model {
                Text("Model: \(model.contract.featureOrder.count) features in contract.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var statusWithFeedbackTabScroll: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: ClinicalTheme.sectionSpacing) {
                privacyBanner
                stitchFatigueResultsStack
                if prediction == nil {
                    Button("Run prediction") {
                        runPrediction()
                    }
                    .buttonStyle(ClinicalPrimaryButtonStyle())
                    .disabled(model == nil || features.isEmpty || loading)
                }
                if features.isEmpty {
                    Text("No prediction payload yet. Open Settings and use Fetch Night, Fetch + Predict, or Load Demo Data.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if let modelLoadError {
                    Text(modelLoadError)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
                if loading { ProgressView().frame(maxWidth: .infinity) }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .scrollIndicators(.visible)
        .scrollContentBackground(.hidden)
        .background(ClinicalTheme.canvasAdaptive)
    }

    private var settingsTabScroll: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: ClinicalTheme.sectionSpacing) {
                settingsHeaderSection
                readinessBanner(onViewAnalysisProfile: {
                    if features.isEmpty {
                        uiMessage = "Load demo data on the Status tab or fetch from Health, then open Settings again."
                    }
                })
                analysisActionsOnly
                if isSimulator {
                    simulatorStitchCard
                }
                if let modelLoadError {
                    Text(modelLoadError)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
                if loading { ProgressView().frame(maxWidth: .infinity) }
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 12)
        }
        .scrollIndicators(.visible)
        .scrollContentBackground(.hidden)
        .background(ClinicalTheme.canvasAdaptive)
    }

    private var settingsHeaderSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Data Settings & Readiness")
                .font(.system(size: 34, weight: .heavy, design: .rounded))
                .foregroundStyle(ClinicalTheme.onSurface)
                .accessibilityAddTraits(.isHeader)
        }
        .padding(.bottom, 2)
    }

    private var historyTabScroll: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                historyTransparencyBanner
                historyHeroTitle
                historyBarChart
                if nightHistory.isEmpty {
                    Text(emptyHistoryHint)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                } else {
                    ForEach(Array(nightHistory.enumerated()), id: \.offset) { index, night in
                        let pred = index < historyPredictions.count ? historyPredictions[index] : nil
                        stitchHistoryRow(night: night, prediction: pred)
                    }
                }
                historyRecoveryInsightCard
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .padding(.bottom, 8)
        }
        .scrollIndicators(.visible)
        .scrollContentBackground(.hidden)
        .background(ClinicalTheme.canvasAdaptive)
    }

    private var historyTransparencyBanner: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: "lock.shield.fill")
                .font(.title3)
                .foregroundStyle(ClinicalTheme.secondaryMuted)
            Text("Research Transparency: All fatigue analysis occurs 100% on-device. Your data never leaves this sanctuary.")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(ClinicalTheme.onSecondaryContainer)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                .fill(ClinicalTheme.secondaryContainer)
        }
    }

    private var historyHeroTitle: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Last 7 Nights Trend")
                .font(.title.weight(.heavy))
                .fontDesign(.rounded)
                .foregroundStyle(ClinicalTheme.onSurface)
            RoundedRectangle(cornerRadius: 2)
                .fill(ClinicalTheme.primary)
                .frame(width: 48, height: 4)
        }
        .padding(.bottom, 4)
    }

    private var historyBarChart: some View {
        let count = min(nightHistory.count, 7)
        let ordered: [(NightSleepSummary, FatiguePrediction?)] = Array(
            (0..<count).map { i -> (NightSleepSummary, FatiguePrediction?) in
                let p = i < historyPredictions.count ? historyPredictions[i] : nil
                return (nightHistory[i], p)
            }.reversed()
        )
        let maxSleep = max(ordered.map { $0.0.features["total_sleep_minutes"] ?? 0 }.max() ?? 1, 1)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .bottom, spacing: 6) {
                ForEach(Array(ordered.enumerated()), id: \.offset) { _, pair in
                    let night = pair.0
                    let mins = night.features["total_sleep_minutes"] ?? 0
                    let h = CGFloat(mins / maxSleep)
                    let isHigh = pair.1?.label == 1
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(isHigh ? ClinicalTheme.errorRed.opacity(0.22) : ClinicalTheme.primary.opacity(0.14))
                        .frame(maxWidth: .infinity)
                        .frame(height: max(36, 140 * h))
                }
            }
            .frame(height: 150)
            if !ordered.isEmpty {
                HStack {
                    ForEach(Array(ordered.enumerated()), id: \.offset) { _, pair in
                        Text(shortHistoryDate(pair.0.nightDate))
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(ClinicalTheme.outlineVariant)
                            .frame(maxWidth: .infinity)
                    }
                }
            }
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(ClinicalTheme.cardFill)
                .shadow(color: .black.opacity(0.05), radius: 6, x: 0, y: 2)
        }
    }

    private func shortHistoryDate(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "M/d"
        return f.string(from: date)
    }

    private func stitchHistoryRow(night: NightSleepSummary, prediction: FatiguePrediction?) -> some View {
        let riskHigh = prediction?.label == 1
        let sleepMin = Int(night.features["total_sleep_minutes"] ?? 0)
        let hr = Int(night.features["hr_mean"] ?? 0)
        let hrv = Int(night.features["hrv_mean"] ?? 0)
        let hStr = "\(sleepMin / 60)h \(sleepMin % 60)m"
        return HStack(spacing: 0) {
            if riskHigh {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(ClinicalTheme.errorRed.opacity(0.45))
                    .frame(width: 4)
                    .padding(.vertical, 6)
                    .padding(.trailing, 12)
            }
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(historyDateLabel(night.nightDate))
                        .font(.caption.weight(.bold))
                        .foregroundStyle(ClinicalTheme.outline)
                    HStack(spacing: 10) {
                        Text(hStr)
                            .font(.title3.weight(.bold))
                            .fontDesign(.rounded)
                        Text(riskHigh ? "HIGH RISK" : "LOW RISK")
                            .font(.system(size: 10, weight: .heavy))
                            .tracking(0.5)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(riskHigh ? ClinicalTheme.errorContainer : ClinicalTheme.primaryFixed)
                            .foregroundStyle(riskHigh ? ClinicalTheme.errorRed : ClinicalTheme.primary)
                            .clipShape(Capsule())
                    }
                }
                Spacer(minLength: 8)
                HStack(spacing: 12) {
                    VStack(spacing: 2) {
                        Text("HR")
                            .font(.caption2)
                            .foregroundStyle(riskHigh ? ClinicalTheme.errorRed.opacity(0.7) : ClinicalTheme.onSurfaceVariant.opacity(0.6))
                        HStack(alignment: .firstTextBaseline, spacing: 2) {
                            Text("\(hr)")
                                .font(.subheadline.weight(.bold))
                            Text("bpm")
                                .font(.caption2)
                        }
                    }
                    Rectangle()
                        .fill(ClinicalTheme.outlineVariant.opacity(0.3))
                        .frame(width: 1, height: 26)
                    VStack(spacing: 2) {
                        Text("HRV")
                            .font(.caption2)
                            .foregroundStyle(riskHigh ? ClinicalTheme.errorRed.opacity(0.7) : ClinicalTheme.onSurfaceVariant.opacity(0.6))
                        HStack(alignment: .firstTextBaseline, spacing: 2) {
                            Text("\(hrv)")
                                .font(.subheadline.weight(.bold))
                            Text("ms")
                                .font(.caption2)
                        }
                    }
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(ClinicalTheme.outlineVariant)
                }
            }
            .padding(.vertical, 6)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(ClinicalTheme.cardFill)
        }
    }

    private func historyDateLabel(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateFormat = "MMM d"
        return f.string(from: date).uppercased()
    }

    private var historyRecoveryInsightCard: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(ClinicalTheme.primaryGradient)
            Circle()
                .fill(Color.white.opacity(0.12))
                .frame(width: 140, height: 140)
                .blur(radius: 24)
                .offset(x: 120, y: 50)
            VStack(alignment: .leading, spacing: 10) {
                Text("Recovery Insight")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.white)
                Text("Based on your last 7 nights, consistency in sleep timing supports recovery. Open Settings after each sync for an updated fatigue estimate.")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.9))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
            Image(systemName: "brain")
                .font(.system(size: 40))
                .foregroundStyle(.white.opacity(0.35))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .padding(20)
        }
        .frame(minHeight: 160)
    }

    /// `fatigue_prediction_results/code.html` composition.
    private var stitchFatigueResultsStack: some View {
        VStack(alignment: .leading, spacing: 22) {
            analysisResultsHeader
            fatigueRiskBentoCard
            stitchLatestNightSummaryCard
            stitchRobustnessPill
            stitchEdgeFooterLine
            stitchResultsPrivacyAside
        }
    }

    private var analysisResultsHeader: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Analysis Results")
                .font(.title2.weight(.heavy))
                .fontDesign(.rounded)
            Spacer()
            Text("Real-time Inference")
                .font(.system(size: 10, weight: .bold))
                .tracking(1.2)
                .foregroundStyle(ClinicalTheme.outline)
        }
        .padding(.horizontal, 2)
    }

    private var fatigueRiskBentoCard: some View {
        let riskHigh = prediction?.label == 1
        let confPct: Int = {
            guard let prediction else { return 0 }
            let c = riskHigh ? prediction.probability1 : prediction.probability0
            return Int((c * 100).rounded())
        }()
        return VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Next-Day Fatigue Risk")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(ClinicalTheme.outline)
                        .textCase(.uppercase)
                    Text(prediction == nil ? "Awaiting prediction" : (riskHigh ? "Critical Alert" : "Within Range"))
                        .font(.system(size: 32, weight: .heavy, design: .rounded))
                        .foregroundStyle(ClinicalTheme.onSurface)
                }
                Spacer()
                if prediction != nil {
                    HStack(spacing: 6) {
                        Image(systemName: riskHigh ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .font(.caption)
                        Text(riskHigh ? "HIGH" : "LOW")
                            .font(.caption.weight(.black))
                            .tracking(1)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(riskHigh ? ClinicalTheme.errorContainer : ClinicalTheme.primaryFixed)
                    .foregroundStyle(riskHigh ? ClinicalTheme.errorRed : ClinicalTheme.primary)
                    .clipShape(Capsule())
                }
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Confidence")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(ClinicalTheme.outline)
                        .textCase(.uppercase)
                    Text(prediction == nil ? "—" : "\(confPct)%")
                        .font(.title2.weight(.bold))
                        .fontDesign(.rounded)
                        .foregroundStyle(ClinicalTheme.primary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(ClinicalTheme.surfaceContainerLow)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Decision Threshold")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(ClinicalTheme.outline)
                        .textCase(.uppercase)
                    Text(
                        model.flatMap { m in
                            m.contract.decisionThreshold.map { String(format: "%.1f", $0) }
                        } ?? "—"
                    )
                        .font(.title2.weight(.bold))
                        .fontDesign(.rounded)
                        .foregroundStyle(ClinicalTheme.onSurface)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(ClinicalTheme.surfaceContainerLow)
                }
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 40, style: .continuous)
                .fill(ClinicalTheme.cardFill)
                .shadow(color: .black.opacity(0.06), radius: 8, x: 0, y: 2)
        }
    }

    private var stitchLatestNightSummaryCard: some View {
        let totalMin = Int(features["total_sleep_minutes"] ?? 0)
        let eff = Int(((features["sleep_efficiency"] ?? 0) * 100).rounded())
        return VStack(alignment: .leading, spacing: 16) {
            Text("Latest Night Summary")
                .font(.headline.weight(.bold))
                .fontDesign(.rounded)
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    HStack(spacing: 14) {
                        ZStack {
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(ClinicalTheme.primaryFixed)
                                .frame(width: 48, height: 48)
                            Image(systemName: "moon.zzz.fill")
                                .font(.title2)
                                .foregroundStyle(ClinicalTheme.primary)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text(formatSleepDuration(totalMin))
                                .font(.title2.weight(.heavy))
                                .fontDesign(.rounded)
                            Text("Total Sleep Duration")
                                .font(.caption.weight(.medium))
                                .foregroundStyle(ClinicalTheme.outline)
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(eff)%")
                            .font(.title2.weight(.bold))
                            .fontDesign(.rounded)
                            .foregroundStyle(ClinicalTheme.primary)
                        Text("Efficiency")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(ClinicalTheme.outline)
                            .textCase(.uppercase)
                    }
                }
                Divider().opacity(0.35)
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                    metricPill("\(formatPct(features["rem_pct"]))%", "REM")
                    metricPill("\(formatPct(features["deep_pct"]))%", "Deep")
                    metricPill("\(formatPct(features["core_pct"]))%", "Core")
                    metricPill("\(Int(features["hr_mean"] ?? 0)) bpm", "HR")
                    metricPill("\(Int(features["hrv_mean"] ?? 0)) ms", "HRV")
                    metricPill("\(Int(features["resp_mean"] ?? 0)) rpm", "Resp")
                }
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: "drop.fill")
                            .font(.caption)
                            .foregroundStyle(ClinicalTheme.secondaryMuted)
                        Text("Oxygen Saturation (SpO₂)")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(ClinicalTheme.onSecondaryContainer)
                    }
                    Spacer()
                    Text("\(formatPct(features["spo2_mean"]))%")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(ClinicalTheme.secondaryMuted)
                }
                .padding(12)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(ClinicalTheme.secondaryContainer.opacity(0.35))
                }
            }
            .padding(18)
            .background {
                RoundedRectangle(cornerRadius: 40, style: .continuous)
                    .fill(ClinicalTheme.cardFill)
                    .shadow(color: .black.opacity(0.05), radius: 6, x: 0, y: 2)
            }
        }
    }

    private func metricPill(_ value: String, _ label: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.headline.weight(.bold))
                .fontDesign(.rounded)
                .minimumScaleFactor(0.8)
                .lineLimit(1)
            Text(label)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(ClinicalTheme.outline)
                .textCase(.uppercase)
        }
        .frame(maxWidth: .infinity)
    }

    private func formatSleepDuration(_ minutes: Int) -> String {
        let h = minutes / 60
        let m = minutes % 60
        return "\(h)h \(m)m"
    }

    private var stitchRobustnessPill: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.shield.fill")
                    .foregroundStyle(ClinicalTheme.primary)
                Text("Data Robustness")
                    .font(.subheadline.weight(.bold))
                    .fontDesign(.rounded)
                    .textCase(.uppercase)
                    .tracking(0.5)
            }
            HStack {
                Text("Missing feature count:")
                    .font(.caption)
                    .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                Spacer()
                Text("\(missingFeatureCount)")
                    .font(.caption.weight(.bold))
            }
            HStack {
                Text("Imputation:")
                    .font(.caption)
                    .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                Spacer()
                Text(missingFeatureCount == 0 ? "None" : "Local median")
                    .font(.caption.weight(.bold))
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: 36, style: .continuous)
                .fill(ClinicalTheme.surfaceContainerHigh)
        }
    }

    private var stitchEdgeFooterLine: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(ClinicalTheme.primary)
                .frame(width: 6, height: 6)
            Text(
                lastInferenceMs.map { ms in
                    "Computed locally in \(format(ms)) ms — no server request."
                } ?? "Run prediction to record on-device inference time — no server request."
            )
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(ClinicalTheme.outline)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
    }

    private var stitchResultsPrivacyAside: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "lock.fill")
                .foregroundStyle(ClinicalTheme.secondaryMuted)
            VStack(alignment: .leading, spacing: 4) {
                Text("Privacy Locked")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(ClinicalTheme.onSecondaryFixedVariant)
                Text("Fatigue analysis is performed using on-device ML accelerators. Your biometric raw data never leaves this terminal.")
                    .font(.system(size: 11))
                    .foregroundStyle(ClinicalTheme.onSecondaryContainer)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                .fill(ClinicalTheme.secondaryContainer.opacity(0.5))
        }
    }

    private var analysisActionsOnly: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Manual Data Actions")
                .font(.system(size: 11, weight: .bold))
                .tracking(1.8)
                .textCase(.uppercase)
                .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                .padding(.bottom, 2)

            Button("Fetch Latest Night") {
                fetchLatestNight()
            }
            .buttonStyle(ClinicalSecondaryButtonStyle(compact: false, fullWidth: true))
            .disabled(!healthKit.authorizationGranted || loading)

            Button {
                fetchLatestNightAndPredict()
            } label: {
                Text("Fetch + Predict")
                    .font(.headline.weight(.bold))
            }
            .buttonStyle(ClinicalPrimaryButtonStyle(compact: false, fullWidth: true))
            .disabled(!healthKit.authorizationGranted || model == nil || loading)

            Button {
                selectedTab = .history
                fetchHistory()
            } label: {
                Label("Open 7-night history", systemImage: "calendar")
            }
            .buttonStyle(ClinicalSecondaryButtonStyle())
            .labelStyle(.titleAndIcon)
            .disabled(loading)
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(ClinicalTheme.surfaceContainerLow)
        }
    }

    private var privacyBanner: some View {
        Label {
            VStack(alignment: .leading, spacing: 4) {
                Text("Privacy Locked")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(ClinicalTheme.onSurface)
                Text("Fatigue analysis and physiological modeling occur 100% on-device. No health data leaves this secure enclave.")
                    .font(.caption)
                    .foregroundStyle(ClinicalTheme.onSurfaceVariant)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } icon: {
            Image(systemName: "lock.shield.fill")
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(ClinicalTheme.primary)
                .font(.title3)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                .fill(ClinicalTheme.secondaryContainer)
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: - Readiness (Stitch `readiness_dashboard`: gradient hero + bento grid)

    private func readinessBanner(onViewAnalysisProfile: @escaping () -> Void) -> some View {
        let availability = healthKit.dataAvailability
        let state = availability.readinessState
        let readinessCount = readinessBentoItems.filter(\.available).count
        let message: String = {
            switch state {
            case .noData:
                return "No sleep data found in HealthKit. Wear your Apple Watch to sleep tonight and check back tomorrow."
            case .collectingBaseline:
                return "Collecting your baseline. You have \(availability.availableHistoryDays) day(s) of history. Personalized predictions available in \(max(0, 7 - availability.availableHistoryDays)) days."
            case .partialPersonalization:
                return "Predictions active. You have \(availability.availableHistoryDays) day(s) of history. Accuracy will continue improving over the next 30 days."
            case .fullPersonalization:
                return "Fully personalized predictions enabled. \(availability.availableHistoryDays) days of your health data establish your unique baseline."
            }
        }()

        return VStack(alignment: .leading, spacing: 20) {
            readinessGradientHero(state: state, message: message, onViewAnalysisProfile: onViewAnalysisProfile)

            HStack {
                Text("Data Readiness")
                    .font(.title3.weight(.bold))
                    .fontDesign(.rounded)
                    .foregroundStyle(ClinicalTheme.onSurface)
                Spacer()
                Text("\(readinessCount)/8 Verified")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(ClinicalTheme.onPrimaryFixed)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(ClinicalTheme.primaryFixed)
                    .clipShape(Capsule())
            }
            .padding(.horizontal, 4)

            LazyVGrid(
                columns: [
                    GridItem(.flexible(), spacing: 16),
                    GridItem(.flexible(), spacing: 16)
                ],
                spacing: 16
            ) {
                ForEach(readinessBentoItems) { item in
                    readinessBentoCell(item)
                }
            }
        }
    }

    private func readinessGradientHero(
        state: ReadinessState,
        message: String,
        onViewAnalysisProfile: @escaping () -> Void
    ) -> some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(ClinicalTheme.primaryGradient)

            Circle()
                .fill(Color.white.opacity(0.12))
                .frame(width: 160, height: 160)
                .blur(radius: 28)
                .offset(x: 140, y: 70)
                .allowsHitTesting(false)

            VStack(alignment: .leading, spacing: 14) {
                Text("Optimization status")
                    .font(.caption2.weight(.bold))
                    .tracking(1.8)
                    .textCase(.uppercase)
                    .foregroundStyle(ClinicalTheme.onPrimaryFixed)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(ClinicalTheme.primaryFixed)
                    .clipShape(Capsule())

                Text(readinessHeroTitle(for: state))
                    .font(.system(size: 28, weight: .heavy, design: .rounded))
                    .foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)

                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.92))
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Button("View analysis profile", action: onViewAnalysisProfile)
                .buttonStyle(ReadinessHeroCTAButtonStyle())
                .padding(.top, 4)
                .accessibilityHint("Scrolls to your latest analysis and metrics when available.")
            }
            .padding(24)
        }
        .shadow(color: .black.opacity(0.14), radius: 16, x: 0, y: 8)
        .accessibilityElement(children: .contain)
    }

    private func readinessHeroTitle(for state: ReadinessState) -> String {
        switch state {
        case .noData: return "Awaiting sleep data"
        case .collectingBaseline: return "Building your baseline"
        case .partialPersonalization: return "Partial personalization"
        case .fullPersonalization: return "Full personalization"
        }
    }

    private var readinessBentoItems: [ReadinessBentoItem] {
        let a = healthKit.dataAvailability
        return [
            ReadinessBentoItem(id: "sleep", title: "Sleep", systemImage: "moon.zzz.fill", available: a.hasSleepData),
            ReadinessBentoItem(id: "hr", title: "HR", systemImage: "heart.fill", available: a.hasHR),
            ReadinessBentoItem(id: "hrv", title: "HRV", systemImage: "waveform.path.ecg", available: a.hasHRV),
            ReadinessBentoItem(id: "resp", title: "Resp", systemImage: "wind", available: a.hasResp),
            ReadinessBentoItem(id: "spo2", title: "SpO₂", systemImage: "drop.fill", available: a.hasSpO2),
            ReadinessBentoItem(id: "steps", title: "Steps", systemImage: "figure.walk", available: a.hasSteps),
            ReadinessBentoItem(id: "energy", title: "Active Energy", systemImage: "bolt.fill", available: a.hasActiveEnergy),
            ReadinessBentoItem(id: "exercise", title: "Exercise Time", systemImage: "timer", available: a.hasWorkout)
        ]
    }

    private func readinessBentoCell(_ item: ReadinessBentoItem) -> some View {
        HStack(spacing: 0) {
            HStack(spacing: 12) {
                Image(systemName: item.systemImage)
                    .font(.title3)
                    .foregroundStyle(item.available ? ClinicalTheme.primary : ClinicalTheme.onSurfaceVariant.opacity(0.5))
                    .frame(width: 28, alignment: .center)
                    .accessibilityHidden(true)
                Text(item.title)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(ClinicalTheme.onSurface)
                    .lineLimit(2)
                    .minimumScaleFactor(0.8)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            Image(systemName: item.available ? "checkmark.circle.fill" : "circle")
                .font(.title3)
                .foregroundStyle(item.available ? ClinicalTheme.primary : ClinicalTheme.onSurfaceVariant.opacity(0.35))
                .accessibilityLabel(item.available ? "\(item.title), available" : "\(item.title), not detected")
        }
        .padding(16)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(ClinicalTheme.cardFill)
                .shadow(color: .black.opacity(0.06), radius: 6, x: 0, y: 2)
        }
        .accessibilityElement(children: .combine)
    }

    /// Stitch `readiness_dashboard` simulator row (`tertiary-container` + chevron).
    private var simulatorStitchCard: some View {
        Button {
            loadSimulatorDemoAndPredict()
        } label: {
            HStack(spacing: 16) {
                Image(systemName: "testtube.2")
                    .font(.title2)
                    .foregroundStyle(ClinicalTheme.onTertiaryContainer.opacity(0.95))
                    .padding(10)
                    .background(ClinicalTheme.onTertiaryContainer.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    Text("Simulator Mode")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(ClinicalTheme.onTertiaryContainer)
                    Text("Load Demo Data + Run Prediction")
                        .font(.caption)
                        .foregroundStyle(ClinicalTheme.onTertiaryContainer.opacity(0.82))
                        .multilineTextAlignment(.leading)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "chevron.right")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(ClinicalTheme.onTertiaryContainer.opacity(0.9))
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(ClinicalTheme.simulatorTint)
            }
        }
        .buttonStyle(.plain)
        .disabled(loading)
        .accessibilityLabel("Simulator mode: load demo data and run prediction")
    }

    // MARK: - Helpers

    private var missingFeatureCount: Int {
        guard let model else { return 0 }
        return model.contract.featureOrder.reduce(0) { partial, feature in
            guard let value = features[feature], value.isFinite else { return partial + 1 }
            return partial
        }
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
                if availability.readinessState == .noData {
                    uiMessage = isSimulator
                        ? "Health access OK, but the Simulator has no sleep data. Use Simulator Mode below or Load Demo Data."
                        : "Health access granted but no sleep data found yet. Wear your watch to sleep tonight!"
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
                    selectedTab = .settings
                    healthKit.checkDataAvailability { _ in }
                case .failure(let error):
                    uiMessage = messageForHealthKitFailure(error, action: "Data fetch failed")
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
                    selectedTab = .settings
                    runPrediction()
                    loading = false
                    uiMessage = "Fetched Apple Health data and predicted on-device."
                case .failure(let error):
                    loading = false
                    uiMessage = messageForHealthKitFailure(error, action: "Fetch + predict failed")
                }
            }
        }
    }

    private func fetchHistory() {
        loading = true
        historyPredictions = []
        healthKit.fetchMultipleNights(count: 7) { result in
            DispatchQueue.main.async {
                loading = false
                switch result {
                case .success(let nights):
                    if nights.isEmpty {
                        if isSimulator {
                            applySyntheticSevenNightHistory(reason: "Simulator: HealthKit returned no nights.")
                        } else {
                            uiMessage = "No sleep nights found in Apple Health for this window."
                        }
                    } else {
                        nightHistory = nights
                        guard let model else { return }
                        historyPredictions = nights.map { night in
                            model.predict(features: model.completeFeatures(night.features))
                        }
                    }
                case .failure(let error):
                    let ns = error as NSError
                    if isSimulator, ns.domain == "HealthKit", ns.code == 5 {
                        applySyntheticSevenNightHistory(reason: "Simulator has no sleep samples in Health.")
                    } else {
                        uiMessage = messageForHealthKitFailure(error, action: "History fetch failed")
                    }
                }
            }
        }
    }

    /// Populates `nightHistory` + `historyPredictions` when HealthKit cannot (typical on Simulator).
    private func applySyntheticSevenNightHistory(reason: String) {
        guard let model else {
            uiMessage = "Model not loaded; cannot build demo trend."
            return
        }
        let cal = Calendar.current
        let todayStart = cal.startOfDay(for: Date())
        let sleepDeltas = [0.0, 25, -35, -130, 40, -20, 55]
        let hrBase = [64, 62, 66, 68, 61, 63, 65]
        var nights: [NightSleepSummary] = []
        for i in 0..<7 {
            guard let date = cal.date(byAdding: .day, value: -i, to: todayStart) else { continue }
            let total = max(260, 410 + sleepDeltas[i])
            let eff = min(0.96, max(0.78, 0.88 + Double(i % 4) * 0.02 - (sleepDeltas[i] < -50 ? 0.08 : 0)))
            let raw: [String: Double] = [
                "total_sleep_minutes": total,
                "sleep_efficiency": eff,
                "hr_mean": Double(hrBase[i]),
                "hrv_mean": Double(58 - i * 3),
                "rem_pct": 0.21,
                "deep_pct": 0.17,
                "core_pct": 0.62
            ]
            let start = cal.date(byAdding: .hour, value: 22, to: date) ?? date
            let end = cal.date(byAdding: .hour, value: 30, to: date) ?? date.addingTimeInterval(8 * 3600)
            nights.append(
                NightSleepSummary(
                    nightDate: date,
                    start: start,
                    end: end,
                    sourceSummary: "Synthetic (demo trend)",
                    features: raw,
                    dataAvailability: healthKit.dataAvailability
                )
            )
        }
        nights.sort { $0.nightDate > $1.nightDate }
        nightHistory = nights
        historyPredictions = nights.map { night in
            model.predict(features: model.completeFeatures(night.features))
        }
        uiMessage = reason + " Showing a demo 7-night trend."
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
        selectedTab = .settings
    }

    private func loadSimulatorDemoAndPredict() {
        _ = ensureModelLoaded()
        loadDemoFeatures()
        selectedTab = .settings
        guard model != nil, !features.isEmpty else { return }
        runPrediction()
    }

    private func loadDemoFeatures() {
        if model == nil {
            _ = ensureModelLoaded()
        }
        guard let model else {
            // Fallback so simulator UX still works even if the contract resource is missing.
            features = [
                "total_sleep_minutes": 415.0,
                "sleep_efficiency": 415.0 / 450.0,
                "rem_pct": 0.22,
                "deep_pct": 0.18,
                "core_pct": 0.60,
                "hr_mean": 64.0,
                "hrv_mean": 58.0,
                "resp_mean": 15.2,
                "spo2_mean": 0.982
            ]
            prediction = nil
            lastSyncAt = Date()
            uiMessage = "Loaded demo features. Model contract missing, so prediction is unavailable."
            if isSimulator {
                nightHistory = []
                historyPredictions = []
            }
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
        if isSimulator {
            applySyntheticSevenNightHistory(reason: "Demo data loaded.")
        }
    }

    @discardableResult
    private func ensureModelLoaded() -> Bool {
        if model != nil { return true }
        do {
            model = try FatigueModel()
            modelLoadError = nil
            return true
        } catch {
            modelLoadError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            uiMessage = "Model contract not loaded."
            return false
        }
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

    /// HealthKitManager reports NSError(domain: "HealthKit", code: …) for empty or invalid samples.
    private func messageForHealthKitFailure(_ error: Error, action: String) -> String {
        let ns = error as NSError
        guard ns.domain == "HealthKit" else {
            return "\(action): \(error.localizedDescription)"
        }
        switch ns.code {
        case 5:
            return "No sleep samples in Apple Health. The Simulator has none—use Load Demo Data. On a real iPhone, wear your Apple Watch to sleep and open the Health app after sync."
        case 6, 7:
            return "Could not build a sleep night from Health data. Try again after your watch has synced."
        default:
            return "\(action): \(error.localizedDescription)"
        }
    }

    private struct ReadinessBentoItem: Identifiable {
        let id: String
        let title: String
        let systemImage: String
        let available: Bool
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
