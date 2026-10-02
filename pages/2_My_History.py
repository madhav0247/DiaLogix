"""Page 2: Patient Assessment History & Trend Tracking.

Displays all previous clinical risk predictions for the logged-in user,
detailed input parameters, historical SHAP factors, and PDF export re-generation.
"""

import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="My History | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import get_current_user, require_auth
from db.database import get_user_predictions
from ml.predict import get_lifestyle_recommendations
from ui_common import (
    apply_custom_theme,
    generate_pdf_report,
    render_disclaimer,
    render_page_header,
)

# 1. Access Control Guard
require_auth(["user", "admin"])
apply_custom_theme()

current_user = get_current_user()

render_page_header(
    title="Assessment History",
    subtitle=f"Review your historical clinical risk scores, feature trends, and previous SHAP attributions.",
)

# Fetch user records
predictions = get_user_predictions(current_user["id"])

if not predictions:
    st.markdown(
        """
        <div class="metric-glass-card" style="text-align: center; padding: 3rem 1.5rem; margin: 2rem 0;">
            <h3 style="color: #f8fafc; margin-bottom: 0.5rem;">No Clinical Assessments Found</h3>
            <p style="color: #94a3b8; max-width: 500px; margin: 0 auto 1.5rem auto;">
                You have not completed any diabetes risk predictions yet. Run your first clinical evaluation to begin tracking your health indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start Your First Risk Assessment", type="primary", use_container_width=False):
        st.switch_page("pages/1_Predict.py")

    render_disclaimer()
    st.stop()


# Historical Stats Overview
total_tests = len(predictions)
latest_pred = predictions[0]
avg_prob = sum(p["probability"] for p in predictions) / total_tests
high_risk_tests = sum(1 for p in predictions if p["risk_band"] == "High")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-glass-card">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">Total Assessments</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #38bdf8; margin-top: 0.2rem;">
                {total_tests}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Lifetime evaluations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    latest_band = latest_pred["risk_band"]
    band_color = "#10b981" if latest_band == "Low" else ("#f59e0b" if latest_band == "Moderate" else "#ef4444")
    badge_class = f"badge-{latest_band.lower()}"
    st.markdown(
        f"""
        <div class="metric-glass-card">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">Latest Risk Score</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: {band_color}; margin-top: 0.2rem;">
                {latest_pred['probability']:.1%}
            </div>
            <span class="{badge_class}">{latest_band} Risk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-glass-card">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">Average Risk Score</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;">
                {avg_prob:.1%}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Across all submissions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-glass-card">
            <span style="color: #94a3b8; font-size: 0.82rem; font-weight: 600;">High Risk Alerts</span>
            <div style="font-size: 1.8rem; font-weight: 800; color: {'#ef4444' if high_risk_tests > 0 else '#10b981'}; margin-top: 0.2rem;">
                {high_risk_tests}
            </div>
            <span style="color: #64748b; font-size: 0.75rem;">Flagged assessments</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Tabular Overview
st.markdown("### Prediction History Log")

table_data = []
for p in predictions:
    table_data.append({
        "ID": f"#{p['id']}",
        "Date & Time": p["created_at"],
        "Risk Band": p["risk_band"],
        "Probability": f"{p['probability']:.1%}",
        "Glucose": p["glucose"],
        "Blood Pressure": p["blood_pressure"],
        "BMI": p["bmi"],
        "Age": int(p["age"]),
    })

history_df = pd.DataFrame(table_data)
st.dataframe(history_df, use_container_width=True, hide_index=True)

# Interactive Inspection Section
st.markdown("---")
st.markdown("### Re-examine Past Assessment & Explainability")

pred_options = {
    f"ID #{p['id']} — {p['created_at']} (Risk: {p['risk_band']} | {p['probability']:.1%})": p["id"]
    for p in predictions
}

selected_label = st.selectbox("Select an assessment to review:", list(pred_options.keys()))
selected_id = pred_options[selected_label]
selected_pred = next(p for p in predictions if p["id"] == selected_id)

detail_col1, detail_col2 = st.columns([1, 1.2])

with detail_col1:
    st.markdown("#### Clinical Input Values")
    inputs_dict = {
        "Pregnancies": selected_pred["pregnancies"],
        "Glucose": selected_pred["glucose"],
        "BloodPressure": selected_pred["blood_pressure"],
        "SkinThickness": selected_pred["skin_thickness"],
        "Insulin": selected_pred["insulin"],
        "BMI": selected_pred["bmi"],
        "DiabetesPedigreeFunction": selected_pred["dpf"],
        "Age": selected_pred["age"],
    }
    for k, v in inputs_dict.items():
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0.6rem; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                <span style="color: #94a3b8; font-size: 0.88rem;">{k}</span>
                <span style="color: #f8fafc; font-weight: 600; font-size: 0.88rem;">{v}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with detail_col2:
    st.markdown("#### Stored SHAP Top Factor Attributions")
    try:
        factors = json.loads(selected_pred["top_factors"]) if selected_pred["top_factors"] else []
    except Exception:
        factors = []

    if factors:
        for f in factors[:5]:
            is_inc = f.get("direction") == "increased"
            icon = "[+]" if is_inc else "[-]"
            col_hex = "#ef4444" if is_inc else "#10b981"
            st.markdown(
                f"""
                <div class="factor-card">
                    <div>
                        <span style="font-weight: 600; color: #f8fafc;">{icon} {f.get('feature')}</span><br/>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Value: <strong>{f.get('value')}</strong></span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-weight: 700; color: {col_hex}; font-size: 0.9rem;">
                            {'+' if is_inc else ''}{f.get('shap_value', 0):.3f}
                        </span><br/>
                        <span style="font-size: 0.75rem; color: #94a3b8;">{f.get('direction', '').upper()}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No detailed SHAP factors stored for this historical entry.")

    # Re-download PDF for historical assessment
    st.markdown("")
    recs = get_lifestyle_recommendations(selected_pred["risk_band"])
    pdf_bytes = generate_pdf_report(
        user_name=current_user["name"],
        patient_email=current_user["email"],
        probability=selected_pred["probability"],
        risk_band=selected_pred["risk_band"],
        inputs=inputs_dict,
        top_factors=factors,
        recommendations=recs,
    )

    st.download_button(
        label=f"Download Report for ID #{selected_pred['id']} (PDF)",
        data=pdf_bytes,
        file_name=f"Assessment_Report_{selected_pred['id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

render_disclaimer()
