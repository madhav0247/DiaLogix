"""Shared UI components, custom styling tokens, and report generators for DiabetesCare AI.

Applies an executive-grade clinical design system based on UI/UX Pro Max guidelines:
- Accessible color palette (WCAG AAA compliant contrast, calm cyan, health green, alert coral)
- Modern typography pairing (Plus Jakarta Sans & Figtree)
- Elevation & glassmorphic depth with subtle micro-animations
- Styled Streamlit widgets (inputs, buttons, tabs, tables, metrics)
- SVG icons and medical disclaimer banners
"""

from typing import Any, Dict, List, Optional
import streamlit as st

CUSTOM_CSS = """
<style>
/* Import High-Legibility Clinical Typography & Material Icons */
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

:root {
    /* Healthcare Design Tokens */
    --color-primary: #0284c7;
    --color-primary-hover: #0369a1;
    --color-primary-glow: rgba(2, 132, 199, 0.35);
    --color-cyan: #06b6d4;
    --color-teal: #0d9488;
    --color-emerald: #10b981;
    --color-amber: #f59e0b;
    --color-rose: #ef4444;

    /* Dark Mode Obsidian & Slate Surfaces */
    --bg-base: #0b0f19;
    --bg-surface: #111827;
    --bg-card: rgba(17, 24, 39, 0.75);
    --bg-card-hover: rgba(30, 41, 59, 0.85);
    --border-subtle: rgba(148, 163, 184, 0.12);
    --border-active: rgba(56, 189, 248, 0.45);
    
    /* Typography Tokens */
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    /* Radii */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-pill: 9999px;
}

/* Global Font Application (excluding Streamlit Material Icons) */
html, body, .stApp, [data-testid="stAppViewContainer"], .main, p, label, input:not([type="checkbox"]):not([type="radio"]) {
    font-family: 'Plus Jakarta Sans', 'Figtree', -apple-system, BlinkMacSystemFont, sans-serif;
}

[class*="css"]:not([data-testid*="Icon"]):not([class*="material-"]):not([translate="no"]),
[class*="st-"]:not([data-testid*="Icon"]):not([class*="material-"]):not([translate="no"]):not([data-testid="stIconMaterial"]) {
    font-family: 'Plus Jakarta Sans', 'Figtree', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ==========================================================================
   Preserve & Enforce Material Symbols / Icons Everywhere
   ========================================================================== */
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
[class*="material-symbols"],
[class*="material-icons"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
span[translate="no"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stExpanderToggleIcon"] span,
button[aria-label*="password"] span,
button[aria-label*="Password"] span,
div[data-baseweb="input"] span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    -webkit-font-smoothing: antialiased !important;
    text-rendering: optimizeLegibility !important;
}

/* Password Visibility Toggle Button Fix */
div[data-baseweb="input"] button[aria-label*="password"],
div[data-baseweb="input"] button[aria-label*="Password"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-muted) !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 0.5rem !important;
    min-width: 2rem !important;
    width: 2.2rem !important;
    height: 2.2rem !important;
}

div[data-baseweb="input"] button[aria-label*="password"]:hover,
div[data-baseweb="input"] button[aria-label*="Password"]:hover {
    color: #38bdf8 !important;
    transform: none !important;
}

/* Sidebar Navigation: Display 'Home' instead of 'app' */
[data-testid="stSidebarNavItems"] li:first-child a[data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"]),
[data-testid="stSidebarNav"] a[href="./"] span:not([data-testid="stIconMaterial"]),
[data-testid="stSidebarNav"] a[href="/"] span:not([data-testid="stIconMaterial"]),
[data-testid="stSidebarNav"] a[href=""] span:not([data-testid="stIconMaterial"]) {
    font-size: 0 !important;
    line-height: 0 !important;
    display: inline-block !important;
}

[data-testid="stSidebarNavItems"] li:first-child a[data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"])::after,
[data-testid="stSidebarNav"] a[href="./"] span:not([data-testid="stIconMaterial"])::after,
[data-testid="stSidebarNav"] a[href="/"] span:not([data-testid="stIconMaterial"])::after,
[data-testid="stSidebarNav"] a[href=""] span:not([data-testid="stIconMaterial"])::after {
    content: "Home" !important;
    font-size: 0.92rem !important;
    line-height: 1.5 !important;
    font-weight: 600 !important;
    visibility: visible !important;
    display: inline-block !important;
}

/* Expander Header & Arrow Symbol Spacing (Prevent Overlap in Delete Account / Danger Zone) */
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    overflow: visible !important;
}

div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
div[data-testid="stExpander"] details summary span[data-testid="stIconMaterial"] {
    flex-shrink: 0 !important;
    margin-right: 0.35rem !important;
    display: inline-flex !important;
    align-items: center !important;
}

/* Base Body & Container Adjustments */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}

/* Streamlit Header / Hamburger Cleanups */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background-color: transparent !important;}

/* ==========================================================================
   Streamlit Native Elements Reskinning
   ========================================================================== */

/* Modern Primary Buttons */
.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #0891b2 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
    cursor: pointer !important;
}

.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #0369a1 0%, #06b6d4 100%) !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.45) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

.stButton > button:active, div.stDownloadButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3) !important;
}

/* Streamlit Forms */
div[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.75rem !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.35) !important;
}

/* Form Inputs (Text, Password, Number) */
div[data-baseweb="input"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
    background-color: rgba(15, 23, 42, 0.95) !important;
}

div[data-baseweb="input"] input {
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* Selectboxes & Dropdowns */
div[data-baseweb="select"] > div {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: var(--radius-md) !important;
    color: #f8fafc !important;
}

/* Streamlit Tabs */
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding-bottom: 4px !important;
}

div[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Metric Display Reskinning */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.35) !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: var(--text-main) !important;
    letter-spacing: -0.02em !important;
}

/* Dataframe & Tables */
div[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f19 0%, #0f172a 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

/* Alerts / Banners */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid transparent !important;
    backdrop-filter: blur(10px) !important;
    padding: 0.85rem 1.25rem !important;
}

/* ==========================================================================
   Custom Healthcare UI Components
   ========================================================================== */

/* Modern Top Header Container */
.portal-header-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}

.portal-header-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0284c7 0%, #38bdf8 50%, #818cf8 100%);
}

.portal-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(135deg, #38bdf8 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.portal-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    margin-bottom: 0;
    line-height: 1.5;
}

/* Glassmorphism Metric Card */
.metric-glass-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.75) 100%);
    border: 1px solid var(--border-subtle);
    backdrop-filter: blur(16px);
    border-radius: var(--radius-lg);
    padding: 1.35rem 1.5rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    position: relative;
}

.metric-glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 10px 30px rgba(2, 132, 199, 0.18);
}

/* Pulsing Status Dot */
.pulsing-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    background-color: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-ring 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
}

@keyframes pulse-ring {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

/* Clinical Risk Badges with Glow */
.badge-low {
    display: inline-flex;
    align-items: center;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: var(--radius-pill);
    padding: 0.35rem 0.95rem;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
}

.badge-moderate {
    display: inline-flex;
    align-items: center;
    background: rgba(245, 158, 11, 0.12);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: var(--radius-pill);
    padding: 0.35rem 0.95rem;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
}

.badge-high {
    display: inline-flex;
    align-items: center;
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: var(--radius-pill);
    padding: 0.35rem 0.95rem;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.2);
}

/* User Role Badges */
.role-pill-admin {
    display: inline-flex;
    align-items: center;
    background: rgba(129, 140, 248, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(129, 140, 248, 0.4);
    border-radius: var(--radius-sm);
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
}

.role-pill-user {
    display: inline-flex;
    align-items: center;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: var(--radius-sm);
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Medical Disclaimer Alert Banner */
.disclaimer-banner {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(245, 158, 11, 0.02) 100%);
    border-left: 4px solid #f59e0b;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    border-top: 1px solid rgba(245, 158, 11, 0.15);
    border-right: 1px solid rgba(245, 158, 11, 0.15);
    border-bottom: 1px solid rgba(245, 158, 11, 0.15);
    padding: 1rem 1.4rem;
    margin: 2rem 0 1rem 0;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.disclaimer-banner strong {
    color: #fbbf24;
}

/* Feature Attribution Item */
.factor-card {
    background: rgba(15, 23, 42, 0.65);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
    padding: 0.85rem 1.15rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s ease;
}

.factor-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
    background: rgba(30, 41, 59, 0.6);
}

/* Clinical Multi-Tier Risk Progress Bar */
.risk-meter-wrapper {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin: 1rem 0;
}

.risk-meter-track {
    height: 12px;
    border-radius: var(--radius-pill);
    background: linear-gradient(90deg, #10b981 0%, #10b981 30%, #f59e0b 30%, #f59e0b 60%, #ef4444 60%, #ef4444 100%);
    position: relative;
    margin: 1.5rem 0 0.5rem 0;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
}

.risk-meter-pointer {
    position: absolute;
    top: -8px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #ffffff;
    border: 3px solid #0f172a;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.6);
    transform: translateX(-50%);
    transition: left 0.5s ease-out;
}
</style>
"""


HIDE_ADMIN_NAV_CSS = """
<style>
/* Hide Admin pages in sidebar navigation for non-admin users */
[data-testid="stSidebarNavItems"] li:nth-child(5),
[data-testid="stSidebarNavItems"] li:nth-child(6),
[data-testid="stSidebarNavItems"] li:nth-child(7),
[data-testid="stSidebarNavItems"] li:has(a[href*="Admin"]),
[data-testid="stSidebarNavItems"] li:has(a[href*="admin"]),
[data-testid="stSidebarNav"] a[href*="Admin"],
[data-testid="stSidebarNav"] a[href*="admin"],
div[data-testid="stSidebarNavLinkContainer"]:has(a[href*="Admin"]),
div[data-testid="stSidebarNavLinkContainer"]:has(a[href*="admin"]) {
    display: none !important;
}
</style>
"""


def apply_custom_theme() -> None:
    """Injects executive-grade CSS styling and typography into the current page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    user = st.session_state.get("user")
    role = user.get("role") if isinstance(user, dict) else st.session_state.get("role")
    is_admin = bool(st.session_state.get("logged_in") and role == "admin")

    if not is_admin:
        st.markdown(HIDE_ADMIN_NAV_CSS, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: str,
    badge_text: Optional[str] = None,
    badge_type: str = "user",
) -> None:
    """Renders a top clinical portal header with user session indicator."""
    apply_custom_theme()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom: 1.4rem;">
                <h1 class="portal-title">{title}</h1>
                <p class="portal-subtitle">{subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        user = st.session_state.get("user")
        if user:
            role = user.get("role", "user")
            role_badge_class = "role-pill-admin" if role == "admin" else "role-pill-user"
            role_label = "Administrator" if role == "admin" else "Patient"

            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 0.6rem;">
                    <div style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 0.3rem;">
                        <span class="pulsing-dot"></span>
                        <span style="color: #f8fafc; font-weight: 700; font-size: 0.95rem;">{user.get('name', 'User')}</span>
                    </div>
                    <span class="{role_badge_class}">{role_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_risk_meter(probability: float, risk_band: str) -> None:
    """Renders a visual horizontal clinical risk stratifier meter."""
    clamped_pct = max(0.0, min(100.0, probability * 100))
    band_color = "#10b981" if risk_band == "Low" else ("#f59e0b" if risk_band == "Moderate" else "#ef4444")
    
    st.markdown(
        f"""
        <div class="risk-meter-wrapper">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">Clinical Risk Spectrum Position</span>
                <span style="font-weight: 800; color: {band_color}; font-size: 1.1rem;">{probability:.1%} ({risk_band})</span>
            </div>
            <div class="risk-meter-track">
                <div class="risk-meter-pointer" style="left: {clamped_pct}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; font-weight: 600; margin-top: 0.3rem;">
                <span style="color: #34d399;">Low (&lt;30%)</span>
                <span style="color: #fbbf24; text-align: center;">Moderate (30% - 60%)</span>
                <span style="color: #f87171; text-align: right;">High (&gt;60%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer() -> None:
    """Renders the standard clinical educational disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; min-width: 4rem;">Notice</div>
            <div>
                <strong>Medical & Educational Disclaimer:</strong><br/>
                This software is developed strictly for educational demonstration and clinical decision support research. It does <strong>not</strong> constitute medical advice or a formal diagnostic tool. Always consult a qualified licensed physician or healthcare specialist for individual health decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_pdf_report(
    user_name: str,
    patient_email: str,
    probability: float,
    risk_band: str,
    inputs: Dict[str, Any],
    top_factors: List[Dict[str, Any]],
    recommendations: List[str],
) -> bytes:
    """Generates an official clinical summary report PDF using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return b"%PDF-1.4 Mock PDF Report"

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 15)
            self.set_text_color(2, 132, 199)
            self.cell(0, 9, "DiabetesCare AI - Clinical Risk Assessment", ln=True, align="C")
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 116, 139)
            self.cell(0, 6, "Explainable Machine Learning Decision Support Report", ln=True, align="C")
            self.set_draw_color(203, 213, 225)
            self.line(10, 26, 200, 26)
            self.ln(5)

        def footer(self):
            self.set_y(-20)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(148, 163, 184)
            self.cell(
                0,
                4,
                "Educational Demonstration Only - Not a Substitute for Medical Advice or Clinical Diagnosis.",
                ln=True,
                align="C",
            )
            self.cell(0, 4, f"Page {self.page_no()}", ln=True, align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Patient Details Section
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, "1. Patient Profile", ln=True)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(95, 6, f"Patient Name: {user_name}", ln=False)
    pdf.cell(95, 6, f"Email: {patient_email}", ln=True)
    pdf.ln(3)

    # Risk Score Summary Box
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Clinical Risk Stratification", ln=True)

    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 22, style="FD")

    pdf.set_y(pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 13)
    if risk_band == "Low":
        pdf.set_text_color(16, 185, 129)
    elif risk_band == "Moderate":
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_text_color(239, 68, 68)

    pdf.cell(95, 7, f"Risk Tier: {risk_band.upper()}", ln=False, align="C")
    pdf.cell(95, 7, f"Predicted Probability: {probability:.1%}", ln=True, align="C")
    pdf.ln(8)

    # Clinical Measurements Table
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Clinical Input Parameters", ln=True)

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(95, 6, " Clinical Indicator", border=1, fill=True)
    pdf.cell(95, 6, " Patient Value", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 8.5)
    for k, v in inputs.items():
        pdf.cell(95, 5.5, f"  {k}", border=1)
        pdf.cell(95, 5.5, f"  {v}", border=1, ln=True)
    pdf.ln(4)

    # Explainability (Top SHAP Contributors)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Explainable AI Factors (SHAP Attribution)", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    for idx, f in enumerate(top_factors[:4], 1):
        direction_text = "increased risk (+)" if f["direction"] == "increased" else "protective / decreased risk (-)"
        pdf.cell(
            0,
            5,
            f"  {idx}. {f['feature']} (value: {f['value']}) -> {direction_text} [impact: {abs(f['shap_value']):.3f}]",
            ln=True,
        )
    pdf.ln(4)

    # Recommendations
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "5. Personalized Preventive Guidance", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    for rec in recommendations:
        pdf.cell(0, 5, f"  - {rec}", ln=True)

    return bytes(pdf.output())
