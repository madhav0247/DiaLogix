"""Page 4: Administrator Analytics Dashboard.

Monitors population-level screening volume, risk band distribution,
high-risk alerts, historical timelines, and average clinical features across all patients.
Strictly restricted to users with the 'admin' role.
"""

import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Admin Dashboard | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import require_auth
from db.database import get_admin_dashboard_stats
from ui_common import apply_custom_theme, render_disclaimer, render_page_header

# 1. Access Control Guard - Strictly Admin Only
require_auth(["admin"])
apply_custom_theme()

render_page_header(
    title="Clinical Analytics & Operations",
    subtitle="Executive overview of patient screening metrics, risk stratification, and data patterns.",
)

# Fetch Aggregated Stats
stats = get_admin_dashboard_stats()

total_users = stats["total_users"]
total_preds = stats["total_predictions"]
high_risk_count = stats["high_risk_count"]
high_risk_pct = stats["high_risk_pct"]
band_dist = stats["band_distribution"]
timeline = stats["timeline"]
averages = stats["averages"]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-glass-card" style="border-top: 4px solid #38bdf8;">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">Registered Users</span>
            <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin: 0.2rem 0;">
                {total_users}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Active accounts across portal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-glass-card" style="border-top: 4px solid #818cf8;">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">Total Predictions</span>
            <div style="font-size: 2.2rem; font-weight: 800; color: #818cf8; margin: 0.2rem 0;">
                {total_preds}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Completed risk calculations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-glass-card" style="border-top: 4px solid #ef4444;">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">High Risk Screenings</span>
            <div style="font-size: 2.2rem; font-weight: 800; color: #ef4444; margin: 0.2rem 0;">
                {high_risk_count}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Requires clinical attention</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-glass-card" style="border-top: 4px solid #f59e0b;">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">High Risk Prevalence</span>
            <div style="font-size: 2.2rem; font-weight: 800; color: #f59e0b; margin: 0.2rem 0;">
                {high_risk_pct}%
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Of all evaluated patients</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Visual Charts Section
chart_col1, chart_col2 = st.columns([1.3, 1])

with chart_col1:
    st.markdown("### Prediction Volume Over Time")
    if timeline:
        timeline_df = pd.DataFrame(timeline)
        timeline_df.columns = ["Date", "Predictions"]
        timeline_df["Date"] = pd.to_datetime(timeline_df["Date"])
        timeline_df = timeline_df.set_index("Date")

        try:
            import plotly.express as px
            fig_timeline = px.line(
                timeline_df.reset_index(),
                x="Date",
                y="Predictions",
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#38bdf8"],
            )
            fig_timeline.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception:
            st.line_chart(timeline_df, use_container_width=True)
    else:
        st.info("No timeline prediction data recorded yet.")

with chart_col2:
    st.markdown("### Risk Tier Distribution")
    if band_dist and sum(band_dist.values()) > 0:
        band_labels = list(band_dist.keys())
        band_counts = list(band_dist.values())
        band_colors = {
            "Low": "#10b981",
            "Moderate": "#f59e0b",
            "High": "#ef4444",
        }
        colors = [band_colors.get(k, "#38bdf8") for k in band_labels]

        try:
            import plotly.graph_objects as go
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=band_labels,
                        values=band_counts,
                        hole=0.6,
                        marker=dict(colors=colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception:
            st.bar_chart(pd.DataFrame(list(band_dist.items()), columns=["Tier", "Count"]).set_index("Tier"))
    else:
        st.info("No risk distribution data available yet.")

st.markdown("---")

# Average Clinical Indicators Row
st.markdown("### Population Mean Clinical Indicators")
st.caption("Calculated across all logged predictions in the SQLite database:")

avg_c1, avg_c2, avg_c3, avg_c4, avg_c5, avg_c6 = st.columns(6)

with avg_c1:
    st.metric("Avg Glucose", f"{averages['avg_glucose']:.1f} mg/dL")
with avg_c2:
    st.metric("Avg Blood Pressure", f"{averages['avg_bp']:.1f} mm Hg")
with avg_c3:
    st.metric("Avg BMI", f"{averages['avg_bmi']:.1f}")
with avg_c4:
    st.metric("Avg Insulin", f"{averages['avg_insulin']:.1f} μU/mL")
with avg_c5:
    st.metric("Avg Age", f"{averages['avg_age']:.1f} yrs")
with avg_c6:
    st.metric("Avg Risk Score", f"{averages['avg_probability']:.1%}")

st.markdown("---")

# Model File Health & System Status
st.markdown("### Model Pipeline Status")
artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

status_items = [
    ("Random Forest Model", os.path.join(artifacts_dir, "model.joblib")),
    ("Imputation Medians", os.path.join(artifacts_dir, "medians.json")),
    ("SHAP TreeExplainer", os.path.join(artifacts_dir, "explainer.joblib")),
    ("Evaluation Metrics", os.path.join(artifacts_dir, "metrics.json")),
    ("Global SHAP Plot", os.path.join(artifacts_dir, "global_shap.png")),
]

s_cols = st.columns(len(status_items))
for col, (name, path) in zip(s_cols, status_items):
    exists = os.path.exists(path)
    with col:
        badge = "Ready" if exists else "Missing"
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.5); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1); text-align: center;">
                <span style="font-size: 0.8rem; color: #94a3b8;">{name}</span>
                <div style="font-weight: 700; font-size: 0.95rem; margin-top: 0.2rem; color: {'#34d399' if exists else '#f87171'};">
                    {badge}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

render_disclaimer()
