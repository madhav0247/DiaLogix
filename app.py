"""DiabetesCare AI - Clinical Portal Entry Point.

Handles user/admin authentication (login and registration), session state routing,
and platform navigation.
"""

import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="DiaLogix - Diabetes Care Portal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth.auth import (
    get_current_user,
    init_session_state,
    is_admin,
    is_logged_in,
    login_user,
    logout,
    register_user,
    set_user_session,
    validate_registration_inputs,
)
from db.database import init_db
from ui_common import apply_custom_theme, render_disclaimer

# Initialize Database Schema & Session State
init_db()
init_session_state()
apply_custom_theme()

# Sidebar branding & session details
with st.sidebar:
    st.markdown("### **DiabetesCare AI**")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.85rem;'>Clinical Risk Assessment with SHAP Explainability</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if is_logged_in():
        user = get_current_user()
        role = user.get("role", "user")
        role_label = "Administrator" if role == "admin" else "Patient"
        role_class = "role-pill-admin" if role == "admin" else "role-pill-user"

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.15);">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">{user.get('name')}</div>
                <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 0.5rem;">{user.get('email')}</div>
                <span class="{role_class}">{role_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        if st.button("Log Out", use_container_width=True):
            logout()
            st.rerun()
    else:
        st.info("Please log in or register to access predictive assessments.")

    st.markdown("---")
    st.caption("AI for Healthcare Project")
    st.caption("Pima Indians Dataset | Random Forest + SHAP")


# Main portal interface
current_user = get_current_user()

if is_logged_in():
    # Authenticated Welcome Dashboard
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 2rem; margin-bottom: 2rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Welcome, {current_user['name']}!
            </h1>
            <p style="color: #cbd5e1; font-size: 1.05rem; margin-top: 0.5rem; max-width: 750px;">
                Your secure clinical portal for transparent diabetes risk screening, predictive probability scoring, and local SHAP factor attribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_user.get("role") == "admin":
        st.subheader("Administrator Navigation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>Analytics Dashboard</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Monitor total users, risk distribution, high-risk cases, and clinical metrics.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open Dashboard", key="btn_admin_dash", use_container_width=True):
                st.switch_page("pages/4_Admin_Dashboard.py")

        with col2:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>User Management</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Manage user accounts, view registrations, activate or deactivate accounts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Manage Users", key="btn_admin_users", use_container_width=True):
                st.switch_page("pages/5_Admin_Users.py")

        with col3:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>All Predictions</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Filter and audit all patient predictions across the system, export to CSV.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View Predictions", key="btn_admin_preds", use_container_width=True):
                st.switch_page("pages/6_Admin_Predictions.py")

    else:
        st.subheader("Patient Services")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>New Risk Assessment</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Input your current health metrics and receive an instant explainable risk score.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Start Assessment", key="btn_user_predict", use_container_width=True):
                st.switch_page("pages/1_Predict.py")

        with col2:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>My History</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Review your past risk evaluations, top factors, and track metrics over time.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View History", key="btn_user_hist", use_container_width=True):
                st.switch_page("pages/2_My_History.py")

        with col3:
            st.markdown(
                """
                <div class="metric-glass-card">
                    <h4>About Model & SHAP</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem;">Learn how the Random Forest classifier works, dataset info, and global SHAP plots.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Explore Model", key="btn_user_about", use_container_width=True):
                st.switch_page("pages/3_About_Model.py")

    render_disclaimer()

else:
    # Unauthenticated Landing & Login/Register
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 2.2rem 2.5rem; margin-bottom: 2rem;">
            <div style="display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 9999px; letter-spacing: 0.05em; margin-bottom: 0.75rem;">
                AI for Healthcare • Decision Support
            </div>
            <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                DiabetesCare AI
            </h1>
            <p style="color: #cbd5e1; font-size: 1.1rem; margin-top: 0.6rem; max-width: 800px; line-height: 1.6;">
                Clinical-grade diabetes risk prediction with patient-centric explainability. Powered by a high-recall Random Forest model trained on the Pima Indians dataset with SHAP local and global attributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        tab_login, tab_register = st.tabs(["Sign In", "Create Patient Account"])

        with tab_login:
            st.markdown("#### Access Your Health Portal")
            with st.form("login_form"):
                login_email = st.text_input("Email Address", placeholder="name@example.com")
                login_password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sign In to Portal", use_container_width=True)

                if submit_login:
                    success, message, user_data = login_user(login_email, login_password)
                    if success and user_data:
                        set_user_session(user_data)
                        st.success(f"Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error(message)

        with tab_register:
            st.markdown("#### Register as a New Patient")
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
                reg_email = st.text_input("Email Address", placeholder="jane@example.com")
                reg_pass = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
                reg_pass2 = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

                if submit_reg:
                    is_valid, err = validate_registration_inputs(reg_name, reg_email, reg_pass, reg_pass2)
                    if not is_valid:
                        st.error(err)
                    else:
                        ok, reg_msg = register_user(reg_name, reg_email, reg_pass, role="user")
                        if ok:
                            st.success(reg_msg)
                            st.info("Switch to the 'Sign In' tab above to log in.")
                        else:
                            st.error(reg_msg)

    with col_right:
        st.markdown(
            """
            <div class="metric-glass-card" style="margin-bottom: 1rem;">
                <h4 style="color: #38bdf8; margin-top: 0;">Key Capabilities</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7; padding-left: 1.2rem;">
                    <li><strong>Recall-Prioritized Model:</strong> High sensitivity to minimize hazardous false negatives in clinical screening.</li>
                    <li><strong>Transparent SHAP Explanations:</strong> Every prediction is paired with a waterfall breakdown showing which factors increased or reduced risk.</li>
                    <li><strong>Rigorous Data Preprocessing:</strong> Automatic zero-handling for physiological variables with training median imputation.</li>
                    <li><strong>Role-Based Access Control:</strong> Isolated patient history and administrative oversight dashboards.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px dashed rgba(148, 163, 184, 0.3); border-radius: 12px; padding: 1rem;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.88rem;">Quick Demo Access:</span>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.3rem;">
                    <strong>Admin Credentials:</strong><br/>
                    Email: <code>admin@diabetescare.ai</code> | Password: <code>Admin@123</code><br/>
                    <em>(Or register a fresh patient account on the left)</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_disclaimer()
