"""Page 3: Model Architecture, Global SHAP Insights, and Clinical Limitations.

Presents transparent documentation of the machine learning pipeline,
evaluation metrics (Recall, ROC-AUC, Confusion Matrix), global SHAP visualizations,
and clinical domain considerations.
"""

import json
import os
import streamlit as st

st.set_page_config(
    page_title="About the Model | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import require_auth
from ui_common import apply_custom_theme, render_disclaimer, render_page_header

# 1. Access Guard
require_auth(["user", "admin"])
apply_custom_theme()

render_page_header(
    title="Model & Explainability Documentation",
    subtitle="Architecture specifications, test-set performance metrics, and global SHAP attribution.",
)

# Load metrics.json if available
artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
metrics_file = os.path.join(artifacts_dir, "metrics.json")
global_shap_img = os.path.join(artifacts_dir, "global_shap.png")

metrics = None
if os.path.exists(metrics_file):
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

# Top Metrics Row
st.markdown("### Held-Out Test Set Performance")

if metrics:
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-glass-card" style="border-top: 4px solid #10b981;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Recall (Sensitivity)</span>
                <div style="font-size: 2rem; font-weight: 800; color: #10b981; margin: 0.2rem 0;">
                    {metrics.get('recall', 0):.1%}
                </div>
                <span style="color: #64748b; font-size: 0.72rem;">PRIORITY CLINICAL METRIC</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-glass-card" style="border-top: 4px solid #38bdf8;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">ROC-AUC Score</span>
                <div style="font-size: 2rem; font-weight: 800; color: #38bdf8; margin: 0.2rem 0;">
                    {metrics.get('roc_auc', 0):.3f}
                </div>
                <span style="color: #64748b; font-size: 0.72rem;">Discrimination capacity</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-glass-card" style="border-top: 4px solid #818cf8;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Accuracy</span>
                <div style="font-size: 2rem; font-weight: 800; color: #818cf8; margin: 0.2rem 0;">
                    {metrics.get('accuracy', 0):.1%}
                </div>
                <span style="color: #64748b; font-size: 0.72rem;">Overall test accuracy</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-glass-card" style="border-top: 4px solid #f59e0b;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Precision</span>
                <div style="font-size: 2rem; font-weight: 800; color: #f59e0b; margin: 0.2rem 0;">
                    {metrics.get('precision', 0):.1%}
                </div>
                <span style="color: #64748b; font-size: 0.72rem;">Positive predictive value</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col5:
        st.markdown(
            f"""
            <div class="metric-glass-card" style="border-top: 4px solid #ec4899;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">F1-Score</span>
                <div style="font-size: 2rem; font-weight: 800; color: #ec4899; margin: 0.2rem 0;">
                    {metrics.get('f1_score', 0):.3f}
                </div>
                <span style="color: #64748b; font-size: 0.72rem;">Harmonic mean</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("Metrics artifact not found yet. Execute `python ml/train.py` to record evaluation scores.")

st.markdown("---")

# Global SHAP Beeswarm & Explainability
st.markdown("### Global Population Feature Attribution (SHAP)")
st.caption(
    "SHAP (SHapley Additive exPlanations) computes game-theoretic Shapley values to reflect how each feature "
    "influences diabetes risk across the entire population."
)

shap_left, shap_right = st.columns([1.5, 1])

with shap_left:
    if os.path.exists(global_shap_img):
        st.image(
            global_shap_img,
            caption="Global SHAP Summary Beeswarm Plot (Held-Out Test Set)",
            use_container_width=True,
        )
    else:
        st.warning("Global SHAP plot will be generated when running `python ml/train.py`.")

with shap_right:
    st.markdown(
        """
        <div class="metric-glass-card">
            <h4 style="color: #38bdf8; margin-top: 0;">Clinical Interpretation of Global SHAP</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; padding-left: 1.2rem;">
                <li><strong>Glucose:</strong> Universally the strongest risk driver. High values (red dots) heavily shift model output into the diabetic probability range.</li>
                <li><strong>BMI (Body Mass Index):</strong> Elevated BMI significantly elevates insulin resistance risk, acting as the secondary global predictor.</li>
                <li><strong>Age & Pedigree:</strong> Older age and stronger genetic family history scores create an elevated baseline risk.</li>
                <li><strong>Protective Factors:</strong> Normal fasting glucose (&lt;100 mg/dL) and normal BMI (&lt;25) provide strong negative (protective) SHAP values.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

render_disclaimer()
