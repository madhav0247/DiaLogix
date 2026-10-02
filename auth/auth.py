"""Authentication and role-based access control module for DiabetesCare AI.

Handles password hashing via bcrypt, user registration, login verification,
and Streamlit session state authentication guards.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import streamlit as st
import bcrypt

from db.database import (
    create_user,
    get_user_by_email,
    get_user_by_id,
)

EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with a secure salt.

    Args:
        password: Plaintext password string.

    Returns:
        Salted bcrypt hash as a UTF-8 string.
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_bytes.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a bcrypt hash.

    Args:
        plain_password: Plaintext password provided during login.
        hashed_password: Stored bcrypt hash string.

    Returns:
        True if passwords match, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def validate_registration_inputs(
    name: str, email: str, password: str, confirm_password: str
) -> Tuple[bool, str]:
    """Validates registration input fields before DB insertion.

    Args:
        name: User's name.
        email: User's email.
        password: User's chosen password.
        confirm_password: Re-entered password for confirmation.

    Returns:
        (is_valid, error_message)
    """
    if not name or len(name.strip()) < 2:
        return False, "Please enter your full name (at least 2 characters)."

    if not email or not re.match(EMAIL_REGEX, email.strip()):
        return False, "Please enter a valid email address."

    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    if password != confirm_password:
        return False, "Passwords do not match."

    return True, ""


def register_user(
    name: str,
    email: str,
    password: str,
    role: str = "user",
) -> Tuple[bool, str]:
    """Registers a new user account after validation and unique email check.

    Args:
        name: Full name.
        email: Email address.
        password: Plaintext password.
        role: 'user' or 'admin' (public registration should always be 'user').

    Returns:
        (success, message)
    """
    clean_email = email.strip().lower()
    existing = get_user_by_email(clean_email)
    if existing:
        return False, "An account with this email address already exists."

    pw_hash = hash_password(password)
    try:
        create_user(
            name=name.strip(),
            email=clean_email,
            password_hash=pw_hash,
            role=role,
        )
        return True, "Registration successful! You may now log in."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"


def login_user(
    email: str, password: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Authenticates a user against database credentials.

    Args:
        email: Login email.
        password: Login password.

    Returns:
        (success, message, user_dict_without_password_hash)
    """
    clean_email = email.strip().lower()
    user = get_user_by_email(clean_email)

    if not user:
        return False, "Invalid email or password.", None

    if not user.get("is_active", 1):
        return (
            False,
            "Your account has been deactivated. Please contact an administrator.",
            None,
        )

    if not verify_password(password, user["password_hash"]):
        return False, "Invalid email or password.", None

    # Return safe user session object
    safe_user = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"],
    }
    return True, "Login successful.", safe_user


# ---------------------------------------------------------------------------
# Streamlit Session State & Guards
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initializes authentication variables in Streamlit session_state."""
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "role" not in st.session_state:
        st.session_state["role"] = None


def set_user_session(user: Dict[str, Any]) -> None:
    """Saves authenticated user profile into session state."""
    st.session_state["user"] = user
    st.session_state["logged_in"] = True
    st.session_state["role"] = user.get("role", "user")


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns currently authenticated user dict or None."""
    init_session_state()
    return st.session_state.get("user")


def is_logged_in() -> bool:
    """Returns True if a user is currently authenticated."""
    init_session_state()
    return bool(st.session_state.get("logged_in")) and st.session_state.get("user") is not None


def is_admin() -> bool:
    """Returns True if the active session has administrator role."""
    init_session_state()
    return is_logged_in() and st.session_state.get("role") == "admin"


def logout() -> None:
    """Clears authentication session state."""
    st.session_state["user"] = None
    st.session_state["logged_in"] = False
    st.session_state["role"] = None


def require_auth(allowed_roles: Optional[List[str]] = None) -> bool:
    """Page guard function. Halts execution if user is unauthenticated or unauthorized.

    Args:
        allowed_roles: List of roles allowed to view page (e.g. ['admin'] or ['user', 'admin']).

    Returns:
        True if authorization passes; otherwise stops page rendering.
    """
    init_session_state()

    if not is_logged_in():
        st.warning("Authentication required. Please log in from the main portal.")
        if st.button("Go to Portal / Log In", key="guard_login_btn"):
            st.switch_page("app.py")
        st.stop()
        return False

    user_role = st.session_state.get("role", "user")

    if allowed_roles and user_role not in allowed_roles:
        st.error("Access Denied. You do not have permission to access this page.")
        st.info(f"Required role: {', '.join(allowed_roles)} | Your role: {user_role}")
        st.stop()
        return False

    return True
