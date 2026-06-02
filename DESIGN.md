---
name: Documentation Minimalist
colors:
  surface: '#f9f9f9'
  surface-dim: '#dadada'
  surface-bright: '#f9f9f9'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f3'
  surface-container: '#eeeeee'
  surface-container-high: '#e8e8e8'
  surface-container-highest: '#e2e2e2'
  on-surface: '#1b1b1b'
  on-surface-variant: '#4c4546'
  inverse-surface: '#303030'
  inverse-on-surface: '#f1f1f1'
  outline: '#7e7576'
  outline-variant: '#cfc4c5'
  surface-tint: '#5e5e5e'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#1b1b1b'
  on-primary-container: '#848484'
  inverse-primary: '#c6c6c6'
  secondary: '#5e5e5e'
  on-secondary: '#ffffff'
  secondary-container: '#e3e2e2'
  on-secondary-container: '#646464'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#1b1b1b'
  on-tertiary-container: '#848484'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c6'
  on-primary-fixed: '#1b1b1b'
  on-primary-fixed-variant: '#474747'
  secondary-fixed: '#e3e2e2'
  secondary-fixed-dim: '#c7c6c6'
  on-secondary-fixed: '#1b1c1c'
  on-secondary-fixed-variant: '#464747'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c6'
  on-tertiary-fixed: '#1b1b1b'
  on-tertiary-fixed-variant: '#474747'
  background: '#f9f9f9'
  on-background: '#1b1b1b'
  surface-variant: '#e2e2e2'
  canvas: '#ffffff'
  ink-deep: '#090909'
  surface-soft: '#fafafa'
  surface-dark: '#171717'
  hairline: '#e5e5e5'
  hairline-strong: '#d4d4d4'
  ink: '#000000'
  charcoal: '#525252'
  mute: '#a3a3a3'
  on-dark-mute: rgba(255, 255, 255, 0.7)
typography:
  display-xl:
    fontFamily: plusJakartaSans
    fontSize: 36px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  display-xl-mobile:
    fontFamily: plusJakartaSans
    fontSize: 30px
    fontWeight: '600'
    lineHeight: '1.2'
  heading-lg:
    fontFamily: plusJakartaSans
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.3'
  body-lg:
    fontFamily: inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: '1.5'
  code-md:
    fontFamily: geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.4'
  label-sm:
    fontFamily: inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: '1'
  caption:
    fontFamily: inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: '1.4'
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  xxs: 2px
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  xxl: 32px
  section: 88px
---

## Overview
Ollama's site is the most aggressively under-designed marketing surface in the AI tooling space, and that is the entire point. The home page reads like a Markdown README rendered with care: a 36px center-aligned heading sits above an inline `curl` install snippet inside a soft-gray pill, a single black "Download" CTA, and a hand-drawn llama mascot as the only ornament. Everything else — automate-your-work block, "Start local. Scale cloud." pricing pair, "Your data stays yours" guarantee strip, FAQ wall on `/pricing` — sits on the same paper-white canvas (`{colors.canvas}`) with quiet `{colors.body}` neutrals carrying the prose. The system is the documentation, and the documentation is the system.
The design philosophy is geometric: every interactive element collapses to `{rounded.full}` (9999px) — buttons, search pills, install-snippet pills, text inputs, and the terminal-traffic-light dots. There are no decorative drop shadows, no gradients, no hero illustrations beyond the llama. Cards (the rare ones, on `/pricing`) use a soft `{rounded.lg}` (12px) and a 1px hairline. The single inverted moment in the entire system is the dark "Max" pricing tier — `{colors.surface-dark}` with white text — which acts as the only attention-grabbing surface in an otherwise studiously flat layout.
Typography pairs SF Pro Rounded (display headings, weight 500–600) with the operating system's default sans (`ui-sans-serif`) for body and `ui-monospace` for code. The roundness of the heading face is the only "personality" the chrome carries — it gently echoes the `{rounded.full}` button geometry without being decorative about it.
**Key Characteristics:**
- Paper-white `{colors.canvas}` end-to-end with no surface alternation — the whole page is one continuous sheet
- Center-aligned hero with `{typography.display-xl}` SF Pro Rounded headline, no eyebrow, no subhead beyond a small "Power OpenClaw with Ollama" line under the llama
- Pill geometry everywhere: every button and pill input is `{rounded.full}`; cards use `{rounded.lg}`; nothing is sharp-cornered except section dividers
- Single-color CTA system: pure black `{colors.primary}` pills carry every action; "Get Pro" / "Get Max" inside pricing cards are the only variations
- Inline `curl` install snippet rendered as a pill with `{typography.code-md}` — the most signature element, sitting directly under the hero headline
- Terminal-mockup card with macOS traffic-light dots and inline `ollama launch openclaw` example — the home page's only "product preview"
- Inverted dark `{component.pricing-card-dark}` for the highest-tier "Max" plan, breaking the flat-white rhythm exactly once per page
## Colors
### Brand & Accent
- **Pure Black** (`{colors.primary}` — `#000000`): the brand. Every primary CTA, every black pill, every link in the nav, and every solid icon. There is no other "brand color."
- **Ink Deep** (`{colors.ink-deep}` — `#090909`): pressed-state black for the primary pill — a single notch below pure.
### Surface
- **Canvas** (`{colors.canvas}` — `#ffffff`): the page itself. Nearly every surface in the system.
- **Soft Surface** (`{colors.surface-soft}` — `#fafafa`): install-snippet pill background, search pill, secondary chip backgrounds, alternating row fill where one is needed.
- **Surface Dark** (`{colors.surface-dark}` — `#171717`): the dark "Max" pricing card and dark CTA strips. The single inverted surface in the system.
- **Hairline** (`{colors.hairline}` — `#e5e5e5`): 1px card border, divider line above footer, divider between FAQ rows.
- **Hairline Strong** (`{colors.hairline-strong}` — `#d4d4d4`): rare slightly stronger divider where extra separation is needed (e.g., between unrelated FAQ groups).
### Text
- **Ink** (`{colors.ink}` — `#000000`): all headlines, primary nav links, button text on light surfaces, prices on pricing cards.
- **Charcoal** (`{colors.charcoal}` — `#525252`): list-item text and disabled-state secondary copy.
- **Body** (`{colors.body}` — `#737373`): default body color for paragraph copy, FAQ answers, footer link text — the system's most-used text color after pure black.
- **Mute** (`{colors.mute}` — `#a3a3a3`): caption text, command-line "comment" gray inside terminal mockups, lowest-emphasis utility text.
- **On Dark** (`{colors.on-dark}` — `#ffffff`): primary text on `{colors.surface-dark}`.
- **On Dark Mute** (`{colors.on-dark-mute}` — `rgba(255,255,255,0.7)`): secondary copy inside the dark "Max" pricing card.
## Typography
### Font Family
- **SF Pro Rounded** (display headings) — Apple's rounded geometric sans, used at weights 500 and 600 for headlines from `{typography.display-xl}` (36px) down to `{typography.heading-lg}` (24px). Falls back to `system-ui` → `-apple-system`.
- **ui-sans-serif** (body, links, buttons, captions) — the operating system's default sans-serif. Carries every non-display text role at 12–20px. Falls back through `system-ui` and platform emoji families.
- **ui-monospace** (code, install snippet, command tags) — the OS default monospace. Used inside the terminal mockup, the inline `curl` install pill, and any inline `<code>` formatting. Falls back to SFMono-Regular → Menlo → Monaco → Consolas.
## Layout
### Spacing System
- **Base unit:** 8px
- **Tokens:** `{spacing.xxs}` (2px) · `{spacing.xs}` (4px) · `{spacing.sm}` (8px) · `{spacing.md}` (12px) · `{spacing.lg}` (16px) · `{spacing.xl}` (24px) · `{spacing.xxl}` (32px) · `{spacing.section}` (88px)
## Shapes
### Border Radius Scale
- `{rounded.none}`: 0px
- `{rounded.sm}`: 6px (code chips)
- `{rounded.md}`: 8px
- `{rounded.lg}`: 12px (cards)
- `{rounded.full}`: 9999px (pills, buttons)
