"""Page 5: Administrator User Management.

Allows searching, inspecting, activating, deactivating, resetting passwords,
and deleting user accounts. Strictly restricted to administrators.
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="User Management | DiabetesCare AI",
    page_icon=None,
    layout="wide",
)

from auth.auth import hash_password, require_auth
from db.database import (
    delete_user,
    list_all_users,
    update_user_password,
    update_user_status,
)
from ui_common import apply_custom_theme, render_disclaimer, render_page_header

# 1. Access Control Guard - Strictly Admin Only
require_auth(["admin"])
apply_custom_theme()

render_page_header(
    title="User Management Directory",
    subtitle="Audit, search, activate/deactivate, or manage credentials for all registered platform accounts.",
)

# Search Bar
search_query = st.text_input(
    "Search Users by Name or Email Address",
    placeholder="Type to filter users...",
)

users = list_all_users(search=search_query)

if not users:
    st.info("No user accounts found matching the search criteria.")
    st.stop()

# Summary count
st.markdown(f"**Found {len(users)} registered account(s):**")

# User Table Overview
table_rows = []
for u in users:
    table_rows.append({
        "ID": u["id"],
        "Name": u["name"],
        "Email": u["email"],
        "Role": u["role"].upper(),
        "Status": "Active" if u["is_active"] else "Inactive",
        "Registered At": u["created_at"],
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### Account Action Console")

user_options = {
    f"ID #{u['id']} — {u['name']} ({u['email']}) [{u['role']}]": u["id"]
    for u in users
}

selected_label = st.selectbox("Select user account to manage:", list(user_options.keys()))
selected_id = user_options[selected_label]
selected_user = next(u for u in users if u["id"] == selected_id)

act_col1, act_col2, act_col3 = st.columns(3)

with act_col1:
    st.markdown("#### Account Status")
    status_label = "Deactivate Account" if selected_user["is_active"] else "Reactivate Account"
    status_msg = "Account is currently ACTIVE." if selected_user["is_active"] else "Account is currently INACTIVE."
    st.caption(status_msg)

    if st.button(status_label, use_container_width=True):
        new_status = 0 if selected_user["is_active"] else 1
        ok = update_user_status(selected_id, new_status)
        if ok:
            st.success("User status successfully updated!")
            st.rerun()
        else:
            st.error("Failed to update status.")

with act_col2:
    st.markdown("#### Password Reset")
    st.caption("Assign a new temporary password:")
    with st.form("reset_pwd_form"):
        new_pwd = st.text_input("New Password", type="password", placeholder="Min 6 characters")
        submit_pwd = st.form_submit_button("Reset Password", use_container_width=True)

        if submit_pwd:
            if not new_pwd or len(new_pwd) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                pw_hash = hash_password(new_pwd)
                ok = update_user_password(selected_id, pw_hash)
                if ok:
                    st.success("Password reset successfully!")
                else:
                    st.error("Failed to update password.")

with act_col3:
    st.markdown("#### Delete Account")
    st.caption("Permanently remove user and cascade delete predictions.")
    with st.expander("Danger Zone: Delete User"):
        st.warning(f"This will permanently delete {selected_user['email']} and all their history.")
        confirm = st.checkbox(f"Confirm deletion of ID #{selected_id}")
        if st.button("Delete User Permanently", type="primary", use_container_width=True, disabled=not confirm):
            ok = delete_user(selected_id)
            if ok:
                st.success("User deleted successfully.")
                st.rerun()
            else:
                st.error("Failed to delete user.")

render_disclaimer()
