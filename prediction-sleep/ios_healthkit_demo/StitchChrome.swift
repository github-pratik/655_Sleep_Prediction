import SwiftUI

// Shell chrome mapped from `stitch_cs655_project/*/code.html` (top header + bottom nav).

enum StitchMainTab: Int, CaseIterable, Hashable {
    case status
    case settings
    case history

    var title: String {
        switch self {
        case .status: return "Status"
        case .settings: return "Settings"
        case .history: return "History"
        }
    }

    var systemImage: String {
        switch self {
        case .status: return "heart.text.square"
        case .settings: return "gearshape"
        case .history: return "chart.line.uptrend.xyaxis"
        }
    }
}

struct StitchTopBar: View {
    var onMenu: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Button(action: onMenu) {
                Image(systemName: "line.3.horizontal")
                    .font(.title3)
                    .foregroundStyle(ClinicalTheme.stitchHeaderTitleTeal)
            }
            .accessibilityLabel("Menu")

            Text("Mobile Computing Demo")
                .font(.title3.weight(.bold))
                .fontDesign(.rounded)
                .foregroundStyle(ClinicalTheme.stitchHeaderTitleTeal)
                .lineLimit(1)
                .minimumScaleFactor(0.75)

            Spacer(minLength: 8)

            Circle()
                .fill(ClinicalTheme.surfaceContainerHigh)
                .frame(width: 40, height: 40)
                .overlay {
                    Image(systemName: "person.fill")
                        .font(.body)
                        .foregroundStyle(ClinicalTheme.secondaryMuted)
                }
                .overlay {
                    Circle()
                        .strokeBorder(Color.white, lineWidth: 2)
                }
                .shadow(color: .black.opacity(0.06), radius: 2, x: 0, y: 1)
                .accessibilityHidden(true)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(ClinicalTheme.stitchHeaderBackground)
    }
}

struct StitchBottomBar: View {
    @Binding var selection: StitchMainTab

    var body: some View {
        HStack(spacing: 4) {
            ForEach(StitchMainTab.allCases, id: \.self) { tab in
                Button {
                    selection = tab
                } label: {
                    VStack(spacing: 4) {
                        Image(systemName: tab.systemImage)
                            .font(.title3)
                            .symbolVariant(selection == tab ? .fill : .none)
                        Text(tab.title.uppercased())
                            .font(.system(size: 10, weight: .semibold))
                            .tracking(1.2)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                    .background {
                        if selection == tab {
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(ClinicalTheme.tabSelectionFill)
                        }
                    }
                    .foregroundStyle(selection == tab ? ClinicalTheme.primary : ClinicalTheme.outlineVariant)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 10)
        .padding(.bottom, 24)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.primary.opacity(0.06))
                .frame(height: 0.5)
        }
    }
}
