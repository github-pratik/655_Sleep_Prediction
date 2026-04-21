import SwiftUI

// Cold-start permissions education (Stitch “Cold Start — Permissions” frame).
// SwiftUI sheet & layout: https://developer.apple.com/documentation/swiftui

struct PermissionOnboardingView: View {
    @Environment(\.dismiss) private var dismiss

    /// Called when the user chooses to open the system Health permission sheet.
    var onRequestAccess: () -> Void

    /// User skipped; still dismisses without requesting.
    var onSkip: () -> Void

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: ClinicalTheme.sectionSpacing) {
                    hero

                    privacyCallout

                    whyWeNeedAccess

                    dataTypesCard

                    simulatorNote
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            .scrollIndicators(.visible)
            .scrollContentBackground(.hidden)
            .background(ClinicalTheme.canvasAdaptive)
            .navigationTitle("Health access")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(ClinicalTheme.canvasAdaptive, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .tint(ClinicalTheme.primary)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Not now") {
                        onSkip()
                        dismiss()
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                bottomActions
            }
        }
    }

    private var hero: some View {
        VStack(alignment: .center, spacing: 10) {
            ZStack {
                Circle()
                    .fill(ClinicalTheme.primary.opacity(0.12))
                    .frame(width: 72, height: 72)
                Image(systemName: "heart.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(ClinicalTheme.primary)
                    .symbolRenderingMode(.hierarchical)
            }
            .frame(maxWidth: .infinity)
            .padding(.top, 8)
            .accessibilityHidden(true)

            Text("Connect Apple Health")
                .font(.largeTitle.bold())
                .fontDesign(.rounded)
                .foregroundStyle(ClinicalTheme.onSurface)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
                .accessibilityAddTraits(.isHeader)

            Text("We read sleep and vitals from Apple Health (including Apple Watch) to summarize your last night and estimate next-day fatigue—entirely on this device.")
                .font(.subheadline)
                .foregroundStyle(ClinicalTheme.secondaryMuted)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
        }
    }

    private var privacyCallout: some View {
        Label {
            VStack(alignment: .leading, spacing: 4) {
                Text("On-device only")
                    .font(.subheadline.weight(.semibold))
                Text("No account. No cloud inference. Predictions run locally after you grant read access.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } icon: {
            Image(systemName: "lock.shield.fill")
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

    private var whyWeNeedAccess: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Why we ask")
                .font(.headline)
                .fontDesign(.rounded)
                .foregroundStyle(ClinicalTheme.onSurface)
            Text("Apple requires a clear purpose for Health data. This demo uses it only to build night-level features (sleep stages, heart rate, HRV, breathing, SpO₂, activity) for the fatigue model.")
                .font(.footnote)
                .foregroundStyle(ClinicalTheme.secondaryMuted)
        }
        .clinicalCard()
    }

    private var dataTypesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Data we read", systemImage: "list.bullet.rectangle.fill")
                .font(.headline)
                .fontDesign(.rounded)
                .foregroundStyle(ClinicalTheme.primary)

            permissionRow("Sleep", "moon.zzz.fill", "Sleep analysis for your most recent night window.")
            permissionRow("Heart rate", "heart.fill", "Nighttime averages for recovery context.")
            permissionRow("HRV", "waveform.path.ecg", "SDNN-style variability when available.")
            permissionRow("Respiratory rate", "lungs.fill", "Breaths per minute summaries.")
            permissionRow("Blood oxygen", "drop.fill", "SpO₂ averages when recorded.")
            permissionRow("Activity", "figure.walk", "Steps and energy to complement sleep signals.")
        }
        .clinicalCard()
    }

    private func permissionRow(_ title: String, _ symbol: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .font(.body)
                .foregroundStyle(ClinicalTheme.primary)
                .frame(width: 24, alignment: .center)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(ClinicalTheme.onSurface)
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(ClinicalTheme.secondaryMuted)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var simulatorNote: some View {
        Group {
#if targetEnvironment(simulator)
            Label {
                Text("Simulator cannot show live Health data. After granting access here, use Load Demo Data on the main screen to try the model.")
                    .font(.footnote)
            } icon: {
                Image(systemName: "info.circle.fill")
                    .foregroundStyle(ClinicalTheme.simulatorTint)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .fill(ClinicalTheme.simulatorTint.opacity(0.12))
            }
            .overlay {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .strokeBorder(ClinicalTheme.simulatorTint.opacity(0.22), lineWidth: 1)
            }
#endif
        }
    }

    private var bottomActions: some View {
        VStack(spacing: 10) {
            Button {
                onRequestAccess()
                dismiss()
            } label: {
                Text("Continue to Health")
            }
            .buttonStyle(ClinicalPrimaryButtonStyle())

            Text("You can change access anytime in Settings → Privacy → Health.")
                .font(.caption2)
                .foregroundStyle(ClinicalTheme.secondaryMuted)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.bar)
    }
}

#Preview {
    PermissionOnboardingView(
        onRequestAccess: {},
        onSkip: {}
    )
}
