"""Page 6: Administrator Global Predictions Audit & CSV Export.

Inspects all clinical risk calculations across all registered patients,
supports multi-dimensional filtering by user, risk band, and date,
and provides CSV audit export.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Audit Predictions | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import require_auth
from db.database import get_all_predictions, list_all_users
from ui_common import apply_custom_theme, render_disclaimer, render_page_header

# 1. Access Control Guard - Strictly Admin Only
require_auth(["admin"])
apply_custom_theme()

render_page_header(
    title="Clinical Predictions Audit",
    subtitle="Query, filter, and export the comprehensive database of patient risk predictions.",
)

# Filter Controls
with st.expander("Filter & Query Parameters", expanded=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        band_filter = st.selectbox(
            "Risk Tier:",
            ["All", "High", "Moderate", "Low"],
        )

    with f_col2:
        all_users = list_all_users()
        user_filter_options = {"All Users": None}
        for u in all_users:
            user_filter_options[f"{u['name']} ({u['email']})"] = u["id"]

        selected_user_label = st.selectbox("Filter by Patient:", list(user_filter_options.keys()))
        filter_user_id = user_filter_options[selected_user_label]

    with f_col3:
        start_date = st.date_input("Start Date", value=None)

    with f_col4:
        end_date = st.date_input("End Date", value=None)

# Query records
predictions = get_all_predictions(
    filter_band=band_filter if band_filter != "All" else None,
    filter_user_id=filter_user_id,
    date_start=start_date.strftime("%Y-%m-%d") if start_date else None,
    date_end=end_date.strftime("%Y-%m-%d") if end_date else None,
)

st.markdown(f"**Found {len(predictions)} matching clinical prediction(s):**")

if not predictions:
    st.info("No predictions match the active filter criteria.")
    render_disclaimer()
    st.stop()

# Build DataFrame
table_rows = []
for p in predictions:
    table_rows.append({
        "ID": p["id"],
        "Patient": p["user_name"],
        "Email": p["user_email"],
        "Risk Tier": p["risk_band"],
        "Risk Probability": f"{p['probability']:.1%}",
        "Glucose": p["glucose"],
        "BP": p["blood_pressure"],
        "BMI": p["bmi"],
        "Insulin": p["insulin"],
        "Age": int(p["age"]),
        "Timestamp": p["created_at"],
    })

df_display = pd.DataFrame(table_rows)

# Actions Row (CSV Export)
act_left, act_right = st.columns([3, 1])
with act_right:
    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Predictions to CSV",
        data=csv_bytes,
        file_name=f"diabetes_predictions_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.dataframe(df_display, use_container_width=True, hide_index=True)

# Inspection Drawer
st.markdown("---")
st.markdown("### Detailed Record Inspection")

pred_map = {
    f"ID #{p['id']} — {p['user_name']} ({p['created_at']}) [Risk: {p['risk_band']} | {p['probability']:.1%}]": p["id"]
    for p in predictions
}

selected_pred_label = st.selectbox("Select prediction to inspect in detail:", list(pred_map.keys()))
sel_id = pred_map[selected_pred_label]
target_pred = next(p for p in predictions if p["id"] == sel_id)

ic1, ic2 = st.columns(2)

with ic1:
    st.markdown("#### Patient & Clinical Metrics")
    st.json({
        "ID": target_pred["id"],
        "Patient Name": target_pred["user_name"],
        "Patient Email": target_pred["user_email"],
        "Risk Band": target_pred["risk_band"],
        "Probability": target_pred["probability"],
        "Binary Class": target_pred["prediction"],
        "Pregnancies": target_pred["pregnancies"],
        "Glucose (mg/dL)": target_pred["glucose"],
        "Blood Pressure (mm Hg)": target_pred["blood_pressure"],
        "Skinfold (mm)": target_pred["skin_thickness"],
        "Insulin (μU/mL)": target_pred["insulin"],
        "BMI": target_pred["bmi"],
        "Diabetes Pedigree Function": target_pred["dpf"],
        "Age": target_pred["age"],
        "Created At": target_pred["created_at"],
    })

with ic2:
    st.markdown("#### Recorded SHAP Top Factors")
    if target_pred.get("top_factors"):
        import json
        try:
            factors_data = json.loads(target_pred["top_factors"])
            st.json(factors_data)
        except Exception:
            st.text(target_pred["top_factors"])
    else:
        st.info("No serialized SHAP factors found.")

render_disclaimer()
