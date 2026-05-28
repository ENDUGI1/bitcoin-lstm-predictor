# ==============================================================================
# QUIET QUANT — Design System
# Single source of truth for tokens, CSS, Plotly themes, helpers, and icons.
# Inspired by Linear, Vercel, and anthropic.com. Restraint over flair.
# ==============================================================================

import streamlit as st


# ==============================================================================
# DESIGN TOKENS (Python-side mirror of CSS variables — for Plotly + helpers)
# ==============================================================================

TOKENS = {
    # Surfaces (cool neutral charcoal scale)
    "surface_sunken":   "#0A0B0C",
    "surface_base":     "#0F1011",
    "surface_elevated": "#16181B",
    "surface_overlay":  "#1E2024",

    # Borders
    "border_subtle":  "rgba(241, 239, 236, 0.06)",
    "border_default": "rgba(241, 239, 236, 0.10)",
    "border_strong":  "rgba(241, 239, 236, 0.18)",

    # Text
    "text_primary":   "#F1EFEC",
    "text_secondary": "#B4B5B6",
    "text_muted":     "#7D8088",
    "text_faint":     "#4F5256",

    # Single accent (clay / burnt sienna — used sparingly)
    "accent":         "#E07856",
    "accent_hover":   "#D26841",
    "accent_active":  "#C45838",
    "accent_soft":    "rgba(224, 120, 86, 0.12)",
    "accent_border":  "rgba(224, 120, 86, 0.32)",

    # Direction tints (desaturated — confidence via shape + position, not loud color)
    "up":          "#9DA59E",
    "up_soft":     "rgba(157, 165, 158, 0.12)",
    "down":        "#A39189",
    "down_soft":   "rgba(163, 145, 137, 0.12)",
    "neutral":     "#7D8088",
    "neutral_soft":"rgba(125, 128, 136, 0.10)",

    # Status (mapped to direction so palette stays small)
    "warning":      "#C9A06E",
    "warning_soft": "rgba(201, 160, 110, 0.12)",
}


# ==============================================================================
# PLOTLY THEME — Quiet Quant chart styling
# Importable dict that every chart in app.py applies via fig.update_layout(**PLOTLY_THEME)
# ==============================================================================

PLOTLY_FONT = "Manrope, -apple-system, system-ui, sans-serif"
PLOTLY_MONO = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font": {
        "family": PLOTLY_FONT,
        "color":  TOKENS["text_muted"],
        "size":   12,
    },
    "margin": {"l": 56, "r": 24, "t": 32, "b": 40},
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor":     TOKENS["surface_overlay"],
        "bordercolor": TOKENS["border_strong"],
        "font": {"family": PLOTLY_MONO, "size": 12, "color": TOKENS["text_primary"]},
    },
    "legend": {
        "orientation": "h",
        "y":           1.04,
        "x":           0.5,
        "xanchor":     "center",
        "yanchor":     "bottom",
        "bgcolor":     "rgba(0,0,0,0)",
        "font":        {"family": PLOTLY_FONT, "size": 11, "color": TOKENS["text_muted"]},
    },
}

PLOTLY_AXIS = {
    "showgrid":   True,
    "gridcolor":  "rgba(241, 239, 236, 0.05)",
    "gridwidth":  1,
    "zeroline":   False,
    "linecolor":  "rgba(241, 239, 236, 0.10)",
    "tickcolor":  "rgba(241, 239, 236, 0.10)",
    "tickfont":   {"family": PLOTLY_MONO, "size": 10, "color": TOKENS["text_muted"]},
    "title":      {"font": {"family": PLOTLY_FONT, "size": 11, "color": TOKENS["text_muted"]}},
}


# ==============================================================================
# INLINE SVG ICONS (replace emoji — vector, themeable via currentColor)
# ==============================================================================

ICONS = {
    "arrow_up":    '<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M5 1L9 6H6V9H4V6H1L5 1Z" fill="currentColor"/></svg>',
    "arrow_down":  '<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M5 9L1 4H4V1H6V4H9L5 9Z" fill="currentColor"/></svg>',
    "arrow_right": '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6H10M10 6L7 3M10 6L7 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "dot":         '<svg width="8" height="8" viewBox="0 0 8 8" fill="none"><circle cx="4" cy="4" r="3" fill="currentColor"/></svg>',
    "check":       '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6L5 9L10 3" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "info":        '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.25"/><path d="M7 6V10M7 4V4.5" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/></svg>',
    "alert":       '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1L13 12H1L7 1Z" stroke="currentColor" stroke-width="1.25" stroke-linejoin="round"/><path d="M7 6V8.5M7 10V10.5" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/></svg>',
    "bolt":        '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M7 1L2 7H6L5 11L10 5H6L7 1Z" fill="currentColor"/></svg>',
    "spark":       '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1L7.5 4.5L11 6L7.5 7.5L6 11L4.5 7.5L1 6L4.5 4.5L6 1Z" fill="currentColor"/></svg>',
    "clock":       '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="5" stroke="currentColor" stroke-width="1.25"/><path d="M6 3.5V6L8 7" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round"/></svg>',
}


# ==============================================================================
# CSS — Quiet Quant
# ==============================================================================

CSS = """
/* ============================================================
   FONTS — Manrope (display + body) + JetBrains Mono (data only)
   Loaded via <link rel="stylesheet"> injected by inject() — see styles.py.
   Not via @import to avoid render-blocking the CSSOM.
   ============================================================ */

/* ============================================================
   DESIGN TOKENS
   ============================================================ */
:root {
    /* Surfaces */
    --qq-surface-sunken:   #0A0B0C;
    --qq-surface-base:     #0F1011;
    --qq-surface-elevated: #16181B;
    --qq-surface-overlay:  #1E2024;

    /* Borders */
    --qq-border-subtle:  rgba(241, 239, 236, 0.06);
    --qq-border-default: rgba(241, 239, 236, 0.10);
    --qq-border-strong:  rgba(241, 239, 236, 0.18);

    /* Text */
    --qq-text-primary:   #F1EFEC;
    --qq-text-secondary: #B4B5B6;
    --qq-text-muted:     #7D8088;
    --qq-text-faint:     #4F5256;

    /* Accent (single — clay) */
    --qq-accent:         #E07856;
    --qq-accent-hover:   #D26841;
    --qq-accent-active:  #C45838;
    --qq-accent-soft:    rgba(224, 120, 86, 0.12);
    --qq-accent-border:  rgba(224, 120, 86, 0.32);

    /* Direction tints */
    --qq-up:        #9DA59E;
    --qq-up-soft:   rgba(157, 165, 158, 0.12);
    --qq-down:      #A39189;
    --qq-down-soft: rgba(163, 145, 137, 0.12);
    --qq-warning:   #C9A06E;
    --qq-warning-soft: rgba(201, 160, 110, 0.12);

    /* Spacing scale (8pt grid) */
    --qq-space-1: 4px;
    --qq-space-2: 8px;
    --qq-space-3: 12px;
    --qq-space-4: 16px;
    --qq-space-5: 20px;
    --qq-space-6: 24px;
    --qq-space-8: 32px;
    --qq-space-10: 40px;
    --qq-space-12: 48px;
    --qq-space-16: 64px;

    /* Radius */
    --qq-radius-sm: 6px;
    --qq-radius-md: 10px;
    --qq-radius-lg: 14px;

    /* Shadow */
    --qq-shadow-sm: 0 1px 2px rgba(0,0,0,0.32);
    --qq-shadow-md: 0 6px 16px rgba(0,0,0,0.36);
    --qq-shadow-lg: 0 16px 36px rgba(0,0,0,0.50);

    /* Motion */
    --qq-ease: cubic-bezier(0.4, 0, 0.2, 1);
    --qq-duration-fast: 120ms;
    --qq-duration-base: 200ms;
    --qq-duration-slow: 350ms;

    /* Type */
    --qq-font-sans: 'Manrope', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    --qq-font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* ============================================================
   GLOBAL RESET + BASE
   ============================================================ */
html, body, [class*="css"], .stApp, .stMarkdown, .stText {
    font-family: var(--qq-font-sans) !important;
    color: var(--qq-text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background-color: var(--qq-surface-base);
    background-image: none;  /* kill the cyan/purple radial blobs */
}

/* Hide Streamlit chrome we don't need — but KEEP the header container visible
   so the sidebar collapse/expand toggle stays accessible. Earlier we hid the
   entire header which broke the "reopen sidebar" button. */
#MainMenu, footer { visibility: hidden; height: 0; }

/* Header: make it visually invisible (transparent, no shadow) but keep its
   children interactive so the sidebar toggle still works. */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Hide the right-side toolbar (Deploy button, hamburger menu, running man)
   but NOT the sidebar collapse control. */
header[data-testid="stHeader"] [data-testid="stToolbar"],
header[data-testid="stHeader"] [data-testid="stStatusWidget"],
header[data-testid="stHeader"] [data-testid="stDecoration"] {
    visibility: hidden;
}

/* Belt-and-braces: guarantee the sidebar collapse/expand controls remain
   visible and clickable across Streamlit versions (the data-testid name has
   changed between 1.30, 1.46, and 1.52). */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[kind="headerNoPadding"] {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    color: var(--qq-text-muted) !important;
    fill: var(--qq-text-muted) !important;
}
[data-testid="stSidebarCollapsedControl"]:hover svg,
[data-testid="stSidebarCollapseButton"]:hover svg {
    color: var(--qq-text-primary) !important;
    fill: var(--qq-text-primary) !important;
}

/* Tabular numerals everywhere */
[class*="Metric"], .stMetric, code, pre {
    font-variant-numeric: tabular-nums;
}

/* ============================================================
   TYPOGRAPHY — hierarchy via weight + size, NEVER gradient
   ============================================================ */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    font-family: var(--qq-font-sans) !important;
    color: var(--qq-text-primary) !important;
    letter-spacing: -0.02em !important;
    text-shadow: none !important;
    background: none !important;
    -webkit-text-fill-color: var(--qq-text-primary) !important;
    -webkit-background-clip: initial !important;
    background-clip: initial !important;
    text-transform: none !important;
    margin: 0 0 var(--qq-space-3) 0 !important;
    line-height: 1.2 !important;
}

.stMarkdown h1 { font-weight: 800 !important; font-size: 36px !important; }
.stMarkdown h2 { font-weight: 700 !important; font-size: 24px !important; }
.stMarkdown h3 { font-weight: 700 !important; font-size: 18px !important; letter-spacing: -0.01em !important; }
.stMarkdown h4 { font-weight: 600 !important; font-size: 15px !important; color: var(--qq-text-muted) !important; }

.stMarkdown p, .stMarkdown li {
    color: var(--qq-text-secondary);
    line-height: 1.6;
    font-size: 14.5px;
}

/* Strong text uses primary color */
.stMarkdown strong { color: var(--qq-text-primary); font-weight: 600; }

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--qq-text-muted) !important;
    font-size: 12.5px !important;
    line-height: 1.5 !important;
}

/* ============================================================
   QQ HELPER CLASSES — used by inline HTML helpers
   ============================================================ */
.qq-eyebrow {
    font-family: var(--qq-font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--qq-text-muted);
    margin: 0;
}

.qq-mono { font-family: var(--qq-font-mono); font-variant-numeric: tabular-nums; }

.qq-card {
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-subtle);
    border-radius: var(--qq-radius-md);
    padding: var(--qq-space-5) var(--qq-space-6);
    box-shadow: var(--qq-shadow-sm);
    transition: border-color var(--qq-duration-base) var(--qq-ease),
                background var(--qq-duration-base) var(--qq-ease);
}
.qq-card:hover { border-color: var(--qq-border-default); }

.qq-card-primary {
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-default);
    border-left: 2px solid var(--qq-accent);
    border-radius: var(--qq-radius-md);
    padding: var(--qq-space-6) var(--qq-space-6);
    box-shadow: var(--qq-shadow-md);
}

/* Section header */
.qq-section-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--qq-space-4);
    margin: var(--qq-space-10) 0 var(--qq-space-5) 0;
    padding-bottom: var(--qq-space-3);
    border-bottom: 1px solid var(--qq-border-subtle);
}
.qq-section-header .qq-title {
    font-family: var(--qq-font-sans);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.015em;
    color: var(--qq-text-primary);
    margin: 0;
}
.qq-section-header .qq-eyebrow { text-align: right; flex-shrink: 0; }

/* Pills */
.qq-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: var(--qq-font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    line-height: 1;
}
.qq-pill-up      { background: var(--qq-up-soft);      color: var(--qq-up);      border: 1px solid rgba(157,165,158,0.24); }
.qq-pill-down    { background: var(--qq-down-soft);    color: var(--qq-down);    border: 1px solid rgba(163,145,137,0.24); }
.qq-pill-neutral { background: rgba(125,128,136,0.10); color: var(--qq-text-muted); border: 1px solid var(--qq-border-default); }
.qq-pill-accent  { background: var(--qq-accent-soft);  color: var(--qq-accent);  border: 1px solid var(--qq-accent-border); }
.qq-pill-warning { background: var(--qq-warning-soft); color: var(--qq-warning); border: 1px solid rgba(201,160,110,0.24); }

/* Live dot — subtle pulse, not blink */
.qq-live-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--qq-accent);
    box-shadow: 0 0 0 0 rgba(224, 120, 86, 0.5);
    animation: qq-pulse 2.4s var(--qq-ease) infinite;
    flex-shrink: 0;
}
@keyframes qq-pulse {
    0%, 100% { box-shadow: 0 0 0 0   rgba(224, 120, 86, 0.5); }
    50%      { box-shadow: 0 0 0 6px rgba(224, 120, 86, 0);   }
}

/* Hero strip (top bar) */
.qq-hero-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--qq-space-3) 0;
    margin-bottom: var(--qq-space-8);
    border-bottom: 1px solid var(--qq-border-subtle);
}
.qq-hero-strip .qq-brand {
    display: flex; align-items: center; gap: var(--qq-space-3);
    font-family: var(--qq-font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--qq-text-muted);
}
.qq-hero-strip .qq-brand img { width: 22px; height: 22px; opacity: 0.9; }
.qq-hero-strip .qq-status {
    display: flex; align-items: center; gap: var(--qq-space-3);
    font-family: var(--qq-font-mono);
    font-size: 11px;
    color: var(--qq-text-muted);
}
.qq-hero-strip .qq-status .qq-divider {
    width: 1px; height: 12px; background: var(--qq-border-default);
}

/* Page header (hero h1) */
.qq-page-header {
    margin: var(--qq-space-2) 0 var(--qq-space-10) 0;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: var(--qq-space-6);
    align-items: end;
}
.qq-page-header .qq-eyebrow { margin-bottom: var(--qq-space-3); }
.qq-page-header h1 {
    font-family: var(--qq-font-sans);
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.1;
    color: var(--qq-text-primary);
    margin: 0;
}
.qq-page-header .qq-subtitle {
    margin-top: var(--qq-space-3);
    font-size: 15px;
    color: var(--qq-text-muted);
    max-width: 60ch;
    line-height: 1.55;
}
.qq-page-header .qq-meta {
    text-align: right;
    font-family: var(--qq-font-mono);
    font-size: 12px;
    color: var(--qq-text-muted);
    line-height: 1.6;
}

/* Big metric value (prediction card hero) */
.qq-metric-hero {
    font-family: var(--qq-font-mono);
    font-size: 48px;
    font-weight: 600;
    letter-spacing: -0.025em;
    color: var(--qq-text-primary);
    line-height: 1.05;
    margin: var(--qq-space-3) 0;
    font-variant-numeric: tabular-nums;
}

/* Arrow row (current -> predicted) */
.qq-arrow-row {
    display: flex;
    align-items: center;
    gap: var(--qq-space-3);
    font-family: var(--qq-font-mono);
    font-size: 13px;
    color: var(--qq-text-muted);
    margin-top: var(--qq-space-2);
}
.qq-arrow-row .qq-arrow { color: var(--qq-text-faint); display: inline-flex; }
.qq-arrow-row .qq-value-to { color: var(--qq-text-primary); }

/* Linear confidence meter */
.qq-meter-track {
    width: 100%;
    height: 6px;
    background: var(--qq-surface-overlay);
    border-radius: 999px;
    overflow: hidden;
    margin: var(--qq-space-3) 0;
}
.qq-meter-fill {
    height: 100%;
    background: var(--qq-accent);
    border-radius: 999px;
    transition: width var(--qq-duration-slow) var(--qq-ease);
}
.qq-meter-fill.qq-low  { background: var(--qq-down); }
.qq-meter-fill.qq-mid  { background: var(--qq-warning); }
.qq-meter-fill.qq-high { background: var(--qq-up); }

/* Scenario row */
.qq-scenario {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--qq-space-2) 0;
    border-bottom: 1px solid var(--qq-border-subtle);
    font-family: var(--qq-font-mono);
    font-size: 13px;
}
.qq-scenario:last-child { border-bottom: none; }
.qq-scenario .qq-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--qq-text-muted);
    font-weight: 500;
}
.qq-scenario .qq-value { color: var(--qq-text-primary); }

/* ============================================================
   STREAMLIT WIDGET OVERRIDES
   ============================================================ */

/* --- METRIC --- */
div[data-testid="stMetric"] {
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-subtle);
    border-radius: var(--qq-radius-md);
    padding: var(--qq-space-5) var(--qq-space-6);
    box-shadow: var(--qq-shadow-sm);
    transition: border-color var(--qq-duration-base) var(--qq-ease);
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--qq-border-default);
}
div[data-testid="stMetric"][data-qq-primary="true"] {
    border-left: 2px solid var(--qq-accent);
}

div[data-testid="stMetricLabel"] {
    font-family: var(--qq-font-mono) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--qq-text-muted) !important;
}
div[data-testid="stMetricLabel"] > * {
    color: var(--qq-text-muted) !important;
}

div[data-testid="stMetricValue"] {
    font-family: var(--qq-font-mono) !important;
    font-size: 30px !important;
    font-weight: 600 !important;
    color: var(--qq-text-primary) !important;
    letter-spacing: -0.02em !important;
    font-variant-numeric: tabular-nums !important;
    text-shadow: none !important;
    line-height: 1.1 !important;
    margin-top: var(--qq-space-2) !important;
}

div[data-testid="stMetricDelta"] {
    font-family: var(--qq-font-mono) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    background: transparent !important;
    margin-top: var(--qq-space-2) !important;
}
/* Streamlit positive delta → up tint */
div[data-testid="stMetricDelta"] svg { display: none !important; }
div[data-testid="stMetricDelta"] [data-testid="stMetricDelta"] { gap: 4px; }

/* --- BUTTONS --- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-default);
    color: var(--qq-text-primary);
    font-family: var(--qq-font-sans);
    font-weight: 500;
    font-size: 13.5px;
    letter-spacing: 0;
    text-transform: none;
    border-radius: var(--qq-radius-sm);
    padding: 9px 16px;
    min-height: 38px;
    transition: all var(--qq-duration-base) var(--qq-ease);
    box-shadow: var(--qq-shadow-sm);
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--qq-surface-overlay);
    border-color: var(--qq-border-strong);
    color: var(--qq-text-primary);
    transform: none;
    box-shadow: var(--qq-shadow-md);
}
div.stButton > button:active { transform: scale(0.99); }
div.stButton > button:focus-visible,
div[data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--qq-accent) !important;
    outline-offset: 2px !important;
}

/* Primary button (st.button type="primary") */
div.stButton > button[kind="primary"] {
    background: var(--qq-accent);
    border: 1px solid var(--qq-accent);
    color: var(--qq-surface-base);
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--qq-accent-hover);
    border-color: var(--qq-accent-hover);
    color: var(--qq-surface-base);
}
div.stButton > button[kind="primary"]:active {
    background: var(--qq-accent-active);
}

/* --- SIDEBAR --- */
section[data-testid="stSidebar"] {
    background-color: var(--qq-surface-sunken);
    border-right: 1px solid var(--qq-border-subtle);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: var(--qq-space-6) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    background: none !important;
    -webkit-text-fill-color: var(--qq-text-primary) !important;
    text-shadow: none !important;
    letter-spacing: -0.01em !important;
    text-transform: none !important;
    font-family: var(--qq-font-sans) !important;
}
section[data-testid="stSidebar"] h1 {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--qq-text-muted) !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--qq-text-muted) !important;
    margin-top: var(--qq-space-6) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--qq-border-subtle) !important;
    margin: var(--qq-space-5) 0 !important;
}

/* --- EXPANDER --- */
div[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--qq-border-subtle);
    border-radius: var(--qq-radius-md);
    overflow: hidden;
    transition: border-color var(--qq-duration-base) var(--qq-ease);
}
div[data-testid="stExpander"]:hover {
    border-color: var(--qq-border-default);
}
div[data-testid="stExpander"] summary {
    padding: var(--qq-space-3) var(--qq-space-4) !important;
    font-size: 13px !important;
    color: var(--qq-text-primary) !important;
    font-weight: 500 !important;
}
div[data-testid="stExpander"] summary:hover { background: var(--qq-surface-elevated); }

/* --- ALERTS --- */
div[data-testid="stAlert"] {
    border-radius: var(--qq-radius-md);
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-subtle);
    border-left-width: 2px;
    padding: var(--qq-space-4) var(--qq-space-5);
    box-shadow: var(--qq-shadow-sm);
}
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--qq-surface-elevated);
}
/* info → muted */
div[data-testid="stNotificationContentInfo"],
div[data-baseweb="notification"][kind="info"] {
    background: var(--qq-surface-elevated) !important;
    border-left-color: var(--qq-text-muted) !important;
    color: var(--qq-text-primary) !important;
}
/* success → up tint */
div[data-testid="stNotificationContentSuccess"],
div[data-baseweb="notification"][kind="success"] {
    background: var(--qq-up-soft) !important;
    border-left-color: var(--qq-up) !important;
    color: var(--qq-text-primary) !important;
}
/* warning → warm */
div[data-testid="stNotificationContentWarning"],
div[data-baseweb="notification"][kind="warning"] {
    background: var(--qq-warning-soft) !important;
    border-left-color: var(--qq-warning) !important;
    color: var(--qq-text-primary) !important;
}
/* error → down tint */
div[data-testid="stNotificationContentError"],
div[data-baseweb="notification"][kind="error"] {
    background: var(--qq-down-soft) !important;
    border-left-color: var(--qq-down) !important;
    color: var(--qq-text-primary) !important;
}

/* --- INPUTS --- */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background: var(--qq-surface-elevated) !important;
    border: 1px solid var(--qq-border-default) !important;
    border-radius: var(--qq-radius-sm) !important;
    transition: border-color var(--qq-duration-base) var(--qq-ease);
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
    border-color: var(--qq-accent) !important;
    box-shadow: 0 0 0 3px var(--qq-accent-soft) !important;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] input {
    color: var(--qq-text-primary) !important;
    font-family: var(--qq-font-sans) !important;
}

/* Date input */
div[data-testid="stDateInput"] input {
    background: var(--qq-surface-elevated) !important;
    border: 1px solid var(--qq-border-default) !important;
    color: var(--qq-text-primary) !important;
    font-family: var(--qq-font-mono) !important;
    font-size: 13px !important;
}

/* Checkbox */
div[data-testid="stCheckbox"] label {
    font-size: 13.5px !important;
    color: var(--qq-text-primary) !important;
}
div[data-testid="stCheckbox"] [data-baseweb="checkbox"] [aria-checked="true"] > div:first-child {
    background: var(--qq-accent) !important;
    border-color: var(--qq-accent) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background: var(--qq-surface-elevated);
    border: 1px solid var(--qq-border-subtle);
    border-radius: var(--qq-radius-md);
    overflow: hidden;
}

/* --- SCROLLBAR --- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--qq-surface-base); }
::-webkit-scrollbar-thumb {
    background: var(--qq-surface-overlay);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: var(--qq-border-strong); }

/* --- PLOTLY (background only — colors set per-chart) --- */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
.js-plotly-plot .legend text { fill: var(--qq-text-muted) !important; }

/* --- LINKS --- */
.stMarkdown a {
    color: var(--qq-accent);
    text-decoration: none;
    border-bottom: 1px solid var(--qq-accent-border);
    transition: border-color var(--qq-duration-base) var(--qq-ease);
}
.stMarkdown a:hover { border-bottom-color: var(--qq-accent); }

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 1024px) {
    .qq-page-header { grid-template-columns: 1fr; }
    .qq-page-header .qq-meta { text-align: left; }
    .qq-page-header h1 { font-size: 32px; }
    .qq-metric-hero { font-size: 38px; }
}
@media (max-width: 768px) {
    div[data-testid="stMetricValue"] { font-size: 24px !important; }
    .qq-metric-hero { font-size: 32px; }
    .qq-hero-strip { flex-direction: column; align-items: flex-start; gap: var(--qq-space-2); }
    .qq-section-header { flex-direction: column; align-items: flex-start; gap: var(--qq-space-2); }
    .qq-section-header .qq-eyebrow { text-align: left; }
}

/* ============================================================
   ACCESSIBILITY — respect motion preference
   ============================================================ */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
    .qq-live-dot { animation: none !important; }
}

/* Focus visible (keyboard nav) */
button:focus-visible, a:focus-visible, [role="button"]:focus-visible,
input:focus-visible, summary:focus-visible {
    outline: 2px solid var(--qq-accent) !important;
    outline-offset: 2px !important;
}
"""


# ==============================================================================
# INJECTION
# ==============================================================================

def inject():
    """Inject the Quiet Quant CSS into the Streamlit app. Call once at startup.

    The @import for Google Fonts can stall paint on some browsers. We bias the
    network with preconnect hints + a <link> tag (which Streamlit's markdown
    renders into the document head via its sanitizer) so fonts begin loading
    in parallel with the first paint, never blocking it.
    """
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="preload" as="style" '
        'href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap">'
        '<link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap">',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ==============================================================================
# HELPER COMPONENTS (return HTML strings — render with st.markdown unsafe_allow)
# ==============================================================================

def eyebrow(text: str) -> str:
    return f'<div class="qq-eyebrow">{text}</div>'


def section_header(title: str, eyebrow_text: str = None) -> str:
    """Render a section divider with optional right-side eyebrow label."""
    right = f'<div class="qq-eyebrow">{eyebrow_text}</div>' if eyebrow_text else ""
    return (
        f'<div class="qq-section-header">'
        f'  <h2 class="qq-title">{title}</h2>'
        f'  {right}'
        f'</div>'
    )


def pill(label: str, variant: str = "neutral", icon: str = None) -> str:
    """variant: up | down | neutral | accent | warning"""
    icon_html = f'<span style="display:inline-flex">{ICONS.get(icon, "")}</span>' if icon else ""
    return f'<span class="qq-pill qq-pill-{variant}">{icon_html}{label}</span>'


def live_dot(label: str = "LIVE") -> str:
    return f'<span style="display:inline-flex; align-items:center; gap:8px"><span class="qq-live-dot"></span><span style="font-family:var(--qq-font-mono);font-size:11px;letter-spacing:0.12em;color:var(--qq-text-muted);">{label}</span></span>'


def hero_strip(brand_label: str, logo_b64: str, status_html: str) -> str:
    """Top sticky-style strip. status_html may contain pills, dots, dividers."""
    return (
        '<div class="qq-hero-strip">'
        f'  <div class="qq-brand">'
        f'    <img src="data:image/svg+xml;base64,{logo_b64}" alt="logo" />'
        f'    <span>{brand_label}</span>'
        f'  </div>'
        f'  <div class="qq-status">{status_html}</div>'
        '</div>'
    )


def page_header(eyebrow_text: str, title: str, subtitle: str, meta_html: str = "") -> str:
    return (
        '<div class="qq-page-header">'
        '  <div>'
        f'    <div class="qq-eyebrow">{eyebrow_text}</div>'
        f'    <h1>{title}</h1>'
        f'    <div class="qq-subtitle">{subtitle}</div>'
        '  </div>'
        f'  <div class="qq-meta">{meta_html}</div>'
        '</div>'
    )


def meter_bar(value_pct: float, segment: str = "auto") -> str:
    """value_pct 0–100. segment: 'low' (<55), 'mid' (55–70), 'high' (>=70), or 'auto'."""
    pct = max(0, min(100, value_pct))
    if segment == "auto":
        if pct >= 70:
            segment = "high"
        elif pct >= 55:
            segment = "mid"
        else:
            segment = "low"
    return (
        f'<div class="qq-meter-track">'
        f'  <div class="qq-meter-fill qq-{segment}" style="width: {pct:.1f}%"></div>'
        f'</div>'
    )
