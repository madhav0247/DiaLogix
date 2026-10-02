"""Admin seeding script for DiabetesCare AI.

Creates the initial administrator account. Supports environment variables,
command line arguments, or defaults.
"""

import argparse
import os
import sys

from auth.auth import hash_password
from db.database import get_user_by_email, create_user, init_db


def seed_admin(name: str, email: str, password: str) -> None:
    """Seeds the initial administrator account in SQLite database."""
    print("=" * 60)
    print("  DiabetesCare AI - Administrator Account Seeding")
    print("=" * 60)

    # Initialize schema
    init_db()

    clean_email = email.strip().lower()
    existing = get_user_by_email(clean_email)
    if existing:
        print(f"[*] Admin account already exists for {clean_email} (ID: {existing['id']}).")
        if existing["role"] != "admin":
            print(f"[!] Warning: User exists but has role '{existing['role']}'.")
        return

    pw_hash = hash_password(password)
    user_id = create_user(
        name=name.strip(),
        email=clean_email,
        password_hash=pw_hash,
        role="admin",
    )
    print(f"[+] Successfully created Administrator account:")
    print(f"    - ID: {user_id}")
    print(f"    - Name: {name}")
    print(f"    - Email: {clean_email}")
    print(f"    - Role: admin")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Admin Account")
    parser.add_argument("--name", default=os.getenv("ADMIN_NAME", "System Administrator"))
    parser.add_argument("--email", default=os.getenv("ADMIN_EMAIL", "admin@diabetescare.ai"))
    parser.add_argument("--password", default=os.getenv("ADMIN_PASSWORD", "Admin@123"))

    args = parser.parse_args()
    seed_admin(args.name, args.email, args.password)
