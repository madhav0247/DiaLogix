"""Page 1: Clinical Diabetes Risk Prediction & Explainability.

Presents a validated clinical input form, runs inference via RandomForestClassifier,
explains the output with local SHAP values, renders custom visual attribution,
and saves the assessment to patient history.
"""

import os
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Risk Prediction | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import get_current_user, require_auth
from db.database import save_prediction
from ml.explain import explain_prediction
from ml.predict import (
    load_model_artifacts,
    predict_diabetes_risk,
    validate_clinical_inputs,
)
from ui_common import (
    apply_custom_theme,
    generate_pdf_report,
    render_disclaimer,
    render_page_header,
    render_risk_meter,
)

# 1. Access Control Guard
require_auth(["user", "admin"])
apply_custom_theme()

current_user = get_current_user()

render_page_header(
    title="Clinical Risk Assessment",
    subtitle="Enter clinical indicators to calculate an explainable risk prediction score.",
)

# Sidebar with quick preset samples
with st.sidebar:
    st.markdown("### Quick Test Profiles")
    st.caption("Load typical patient profiles to quickly test model sensitivity:")

    preset = st.radio(
        "Choose a test profile:",
        ["Custom", "High Risk Sample", "Moderate Risk Sample", "Healthy / Low Risk"],
        index=0,
    )

    preset_values = {
        "High Risk Sample": {
            "preg": 6, "glu": 168.0, "bp": 82.0, "skin": 35.0,
            "ins": 220.0, "bmi": 38.2, "dpf": 0.85, "age": 52
        },
        "Moderate Risk Sample": {
            "preg": 3, "glu": 125.0, "bp": 74.0, "skin": 26.0,
            "ins": 110.0, "bmi": 28.5, "dpf": 0.45, "age": 38
        },
        "Healthy / Low Risk": {
            "preg": 1, "glu": 88.0, "bp": 68.0, "skin": 20.0,
            "ins": 75.0, "bmi": 22.4, "dpf": 0.22, "age": 24
        }
    }

# Check if model artifact exists
artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
model_path = os.path.join(artifacts_dir, "model.joblib")

if not os.path.exists(model_path):
    st.error(
        "Trained model artifacts not found! An administrator or developer must execute "
        "`python ml/train.py` first to train the model and generate artifacts."
    )
    st.info("Tip: You can run training from the command line or terminal.")
    render_disclaimer()
    st.stop()


# Cache model and medians loading
@st.cache_resource(show_spinner="Loading predictive models & imputation weights...")
def get_cached_artifacts():
    return load_model_artifacts()


try:
    model, medians = get_cached_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# Preset values helper
pv = preset_values.get(preset, {})

# Clinical Input Form
with st.form("clinical_prediction_form"):
    st.markdown("### Patient Clinical Measurements")
    st.caption("Note: Enter '0' for Skinfold Thickness or Insulin if measurement is unavailable (will be imputed via training medians).")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=pv.get("preg", 1),
            step=1,
            help="Total number of times pregnant (0 to 20).",
        )
        glucose = st.number_input(
            "Plasma Glucose (mg/dL)",
            min_value=40.0,
            max_value=300.0,
            value=float(pv.get("glu", 115.0)),
            step=1.0,
            help="2-hour oral glucose tolerance test level (normal: 70-140 mg/dL).",
        )

    with col2:
        blood_pressure = st.number_input(
            "Diastolic BP (mm Hg)",
            min_value=30.0,
            max_value=200.0,
            value=float(pv.get("bp", 72.0)),
            step=1.0,
            help="Diastolic blood pressure (normal: 60-80 mm Hg).",
        )
        skin_thickness = st.number_input(
            "Triceps Skinfold (mm)",
            min_value=0.0,
            max_value=100.0,
            value=float(pv.get("skin", 23.0)),
            step=1.0,
            help="Triceps skin fold thickness in mm. Enter 0 if unknown.",
        )

    with col3:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=0.0,
            max_value=900.0,
            value=float(pv.get("ins", 0.0)),
            step=1.0,
            help="2-hour serum insulin. Enter 0 if unknown.",
        )
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=70.0,
            value=float(pv.get("bmi", 28.0)),
            step=0.1,
            help="Weight in kg / (height in m)^2.",
        )

    with col4:
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.05,
            max_value=2.50,
            value=float(pv.get("dpf", 0.38)),
            step=0.01,
            format="%.3f",
            help="Genetic risk score based on family diabetes history (0.05 to 2.50).",
        )
        age = st.number_input(
            "Age (Years)",
            min_value=1,
            max_value=120,
            value=int(pv.get("age", 35)),
            step=1,
            help="Patient age in completed years.",
        )

    submit_button = st.form_submit_button(
        "Calculate Explainable Risk Prediction",
        use_container_width=True,
    )

if submit_button:
    raw_inputs = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    # Input validation
    is_valid, validation_errors = validate_clinical_inputs(raw_inputs)
    if not is_valid:
        st.error("Please correct the following input errors before proceeding:")
        for err in validation_errors:
            st.warning(f"• {err}")
    else:
        with st.spinner("Processing clinical features & generating SHAP explanation..."):
            # 1. Run inference
            result = predict_diabetes_risk(raw_inputs, model=model, medians=medians)
            prob = result["probability"]
            band = result["risk_band"]
            color = result["risk_color"]
            badge_class = result["badge_class"]
            recs = result["recommendations"]

            # 2. Compute local SHAP explanation
            explanation = explain_prediction(
                model=model,
                processed_df=result["processed_df"],
                probability=prob,
            )
            top_factors = explanation["top_factors"]
            narrative = explanation["summary_narrative"]
            fig = explanation["figure"]

            # 3. Persist prediction to database
            pred_id = save_prediction(
                user_id=current_user["id"],
                inputs=raw_inputs,
                probability=prob,
                prediction=result["prediction"],
                risk_band=band,
                top_factors=top_factors,
            )

        st.success(f"Assessment complete! Stored in patient history (ID: #{pred_id}).")

        # -------------------------------------------------------------------
        # Results Section
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Prediction & Risk Stratification")

        # Interactive Risk Position Meter
        render_risk_meter(prob, band)

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.2])

        with res_col1:
            st.markdown(
                f"""
                <div class="metric-glass-card" style="text-align: center; border-left: 6px solid {color}; box-shadow: 0 0 25px {color}22;">
                    <span style="color: #94a3b8; font-size: 0.88rem; font-weight: 600; text-transform: uppercase;">Estimated Risk Probability</span>
                    <div style="font-size: 3rem; font-weight: 800; color: {color}; margin: 0.4rem 0;">
                        {prob:.1%}
                    </div>
                    <span class="{badge_class}">{band} Risk Tier</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            pred_label = "Elevated Risk Detected" if result["prediction"] == 1 else "Low Risk Profile"
            pred_desc = "Patient profile exhibits indicators consistent with diabetic risk." if result["prediction"] == 1 else "Patient profile does not currently indicate high diabetic risk."
            st.markdown(
                f"""
                <div class="metric-glass-card">
                    <span style="color: #94a3b8; font-size: 0.88rem; font-weight: 600;">Model Classification</span>
                    <h3 style="margin: 0.4rem 0; font-size: 1.25rem; color: #f8fafc;">{pred_label}</h3>
                    <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0;">{pred_desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col3:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <span style="color: #94a3b8; font-size: 0.88rem; font-weight: 600;">Clinical Risk Bands</span>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                        <div><strong>Low Risk:</strong> &lt; 30% probability</div>
                        <div><strong>Moderate Risk:</strong> 30% – 60% probability</div>
                        <div><strong>High Risk:</strong> &gt; 60% probability</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------------
        # Explainability Section (SHAP)
        # -------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Transparent Decision Breakdown (SHAP)")
        st.caption("How each clinical measurement pushed your estimated risk score up or down.")

        if narrative:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; border-radius: 0 10px 10px 0; padding: 1rem 1.25rem; margin-bottom: 1.25rem; color: #f1f5f9; font-size: 0.95rem;">
                    <strong>Clinical Translation:</strong> {narrative}
                </div>
                """,
                unsafe_allow_html=True,
            )

        shap_col1, shap_col2 = st.columns([1.6, 1])

        with shap_col1:
            st.pyplot(fig)
            plt.close(fig)

        with shap_col2:
            st.markdown("#### Primary Drivers")
            for f in top_factors[:4]:
                is_inc = f["direction"] == "increased"
                icon = "[+]" if is_inc else "[-]"
                impact_color = "#ef4444" if is_inc else "#10b981"
                st.markdown(
                    f"""
                    <div class="factor-card">
                        <div>
                            <span style="font-weight: 600; color: #f8fafc;">{icon} {f['feature']}</span><br/>
                            <span style="font-size: 0.8rem; color: #94a3b8;">Value: <strong>{f['value']}</strong></span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-weight: 700; color: {impact_color}; font-size: 0.9rem;">
                                {'+' if is_inc else ''}{f['shap_value']:.3f}
                            </span><br/>
                            <span style="font-size: 0.75rem; color: #94a3b8;">{f['direction'].upper()}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------------
        # Recommendations & Export
        # -------------------------------------------------------------------
        st.markdown("---")
        rec_col1, rec_col2 = st.columns([2, 1])

        with rec_col1:
            st.markdown("### Lifestyle & Preventative Guidance")
            for r in recs:
                st.markdown(f"- {r}")

        with rec_col2:
            st.markdown("### Patient Report")
            st.caption("Download an official clinical summary document for personal records or physician consultation.")

            pdf_bytes = generate_pdf_report(
                user_name=current_user["name"],
                patient_email=current_user["email"],
                probability=prob,
                risk_band=band,
                inputs=raw_inputs,
                top_factors=top_factors,
                recommendations=recs,
            )

            st.download_button(
                label="Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"Diabetes_Risk_Assessment_{current_user['name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

render_disclaimer()
