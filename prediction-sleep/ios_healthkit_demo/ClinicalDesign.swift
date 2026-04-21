import SwiftUI

// SwiftUI: https://developer.apple.com/documentation/swiftui
//
// Visual tokens: Stitch “Precision Sanctuary” — canonical hex lives in Tailwind config inside:
//   prediction-sleep/stitch_cs655_project/*/code.html
// Narrative: stitch_cs655_project/circadian_clarity/DESIGN.md
// Index: stitch_cs655_project/README.md

enum ClinicalTheme {
    // MARK: Stitch light palette (named colors from design system)

    static let primary = Color(red: 0 / 255, green: 101 / 255, blue: 101 / 255) // #006565
    static let primaryContainer = Color(red: 0 / 255, green: 128 / 255, blue: 128 / 255) // #008080
    /// `primary_fixed` #93f2f2 — status pill on gradient hero (`readiness_dashboard` Stitch)
    static let primaryFixed = Color(red: 147 / 255, green: 242 / 255, blue: 242 / 255)
    /// `on_primary_fixed` #002020
    static let onPrimaryFixed = Color(red: 0 / 255, green: 32 / 255, blue: 32 / 255)
    static let secondaryMuted = Color(red: 72 / 255, green: 98 / 255, blue: 115 / 255) // #486273
    /// `on-secondary-container` #4e6879
    static let onSecondaryContainer = Color(red: 78 / 255, green: 104 / 255, blue: 121 / 255)
    /// `on-secondary-fixed-variant` #304a5a
    static let onSecondaryFixedVariant = Color(red: 48 / 255, green: 74 / 255, blue: 90 / 255)

    /// Canvas `surface` #f7f9fc
    static let canvas = Color(red: 247 / 255, green: 249 / 255, blue: 252 / 255)
    /// Mid tier `surface_container_low` #f2f4f7 (inset strips)
    static let surfaceContainerLow = Color(red: 242 / 255, green: 244 / 255, blue: 247 / 255)
    /// `surface-container-high` #e6e8eb (Stitch top bar avatar / chips)
    static let surfaceContainerHigh = Color(red: 230 / 255, green: 232 / 255, blue: 235 / 255)
    /// Transparency banner `secondary_container` #cbe6fb
    static let secondaryContainer = Color(red: 203 / 255, green: 230 / 255, blue: 251 / 255)
    /// Body text `on_surface` #191c1e
    static let onSurface = Color(red: 25 / 255, green: 28 / 255, blue: 30 / 255)
    /// Muted labels `on-surface-variant` #3e4949 (Stitch tailwind-config)
    static let onSurfaceVariant = Color(red: 62 / 255, green: 73 / 255, blue: 73 / 255)
    /// Simulator / caution `tertiary_container` #b25b00 on light bg
    static let simulatorTint = Color(red: 178 / 255, green: 91 / 255, blue: 0 / 255)
    /// `on-tertiary-container` #fff8f5 (Stitch readiness simulator row)
    static let onTertiaryContainer = Color(red: 255 / 255, green: 248 / 255, blue: 245 / 255)
    /// `outline-variant` #bdc9c8
    static let outlineVariant = Color(red: 189 / 255, green: 201 / 255, blue: 200 / 255)
    /// `outline` #6e7979
    static let outline = Color(red: 110 / 255, green: 121 / 255, blue: 121 / 255)

    static let cardCorner: CGFloat = 14
    /// Vertical gap between major Stitch “frames”
    static let sectionSpacing: CGFloat = 24
    /// Inner stack spacing inside a section
    static let cardSpacing: CGFloat = 16
    static let cardPadding: CGFloat = 18

    /// Slightly richer than a two-stop blend so the hero reads as a gradient on device (Stitch `from-primary to-primary-container`).
    static let primaryGradient = LinearGradient(
        colors: [primary, Color(red: 0 / 255, green: 112 / 255, blue: 112 / 255), primaryContainer],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    /// Card on canvas: white in light, grouped secondary in dark.
    static var cardFill: Color {
        Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? .secondarySystemGroupedBackground
                : .white
        })
    }

    static var canvasAdaptive: Color {
        Color(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark
                ? .systemGroupedBackground
                : UIColor(red: 247 / 255, green: 249 / 255, blue: 252 / 255, alpha: 1)
        })
    }

    /// Stitch bottom tab active pill (`teal-50`).
    static let tabSelectionFill = Color(red: 240 / 255, green: 253 / 255, blue: 250 / 255)
    static let stitchHeaderBackground = Color(red: 241 / 255, green: 245 / 255, blue: 249 / 255)
    static let stitchHeaderTitleTeal = Color(red: 17 / 255, green: 94 / 255, blue: 89 / 255)
    static let errorRed = Color(red: 186 / 255, green: 26 / 255, blue: 26 / 255)
    /// `error-container` #ffdad6
    static let errorContainer = Color(red: 255 / 255, green: 218 / 255, blue: 214 / 255)
}

// MARK: - Cards (Stitch: white sheets on soft canvas, soft lift — no hard 1pt dividers)

struct ClinicalCardStyle: ViewModifier {
    var elevated: Bool = true

    func body(content: Content) -> some View {
        content
            .padding(ClinicalTheme.cardPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .fill(ClinicalTheme.cardFill)
                    .shadow(
                        color: elevated ? Color.black.opacity(0.06) : .clear,
                        radius: elevated ? 12 : 0,
                        x: 0,
                        y: elevated ? 4 : 0
                    )
            }
    }
}

struct ClinicalInsetStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(ClinicalTheme.cardPadding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .fill(ClinicalTheme.surfaceContainerLow)
            }
    }
}

extension View {
    func clinicalCard(elevated: Bool = true) -> some View {
        modifier(ClinicalCardStyle(elevated: elevated))
    }

    func clinicalInset() -> some View {
        modifier(ClinicalInsetStyle())
    }
}

// MARK: - Buttons (Stitch: gradient primary CTA)

struct ClinicalPrimaryButtonStyle: ButtonStyle {
    /// Tighter vertical padding for toolbars in a row.
    var compact: Bool = false
    /// When true, gradient background expands (stacked CTAs and equal-width rows).
    var fullWidth: Bool = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .padding(.vertical, compact ? 11 : 14)
            .padding(.horizontal, compact ? 8 : 16)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(ClinicalTheme.primaryGradient)
            }
            .shadow(color: Color.black.opacity(0.14), radius: 5, x: 0, y: 2)
            .opacity(configuration.isPressed ? 0.9 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}

struct ClinicalSecondaryButtonStyle: ButtonStyle {
    var compact: Bool = false
    var fullWidth: Bool = true

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(ClinicalTheme.primary)
            .multilineTextAlignment(.center)
            .padding(.vertical, compact ? 11 : 14)
            .padding(.horizontal, compact ? 8 : 16)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(ClinicalTheme.cardFill)
                    .overlay {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(ClinicalTheme.outlineVariant, lineWidth: 1)
                    }
            }
            .opacity(configuration.isPressed ? 0.85 : 1)
    }
}

/// White CTA on gradient readiness hero (Stitch `readiness_dashboard` “View Analysis Profile”).
struct ReadinessHeroCTAButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.bold))
            .foregroundStyle(ClinicalTheme.primary)
            .padding(.horizontal, 22)
            .padding(.vertical, 11)
            .background {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.white)
                    .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
            }
            .opacity(configuration.isPressed ? 0.9 : 1)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
    }
}

/// Primary CTA on `cold_start_permissions` connection card (`shadow-primary/20`).
struct ColdStartPrimaryCTAButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline.weight(.bold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background {
                RoundedRectangle(cornerRadius: ClinicalTheme.cardCorner, style: .continuous)
                    .fill(ClinicalTheme.primary)
                    .shadow(color: ClinicalTheme.primary.opacity(0.28), radius: 12, x: 0, y: 5)
            }
            .opacity(configuration.isPressed ? 0.9 : 1)
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}
