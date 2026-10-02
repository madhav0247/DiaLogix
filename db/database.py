"""SQLite database connection and query helpers for DiabetesCare AI.

Provides parameterized CRUD operations for users and predictions,
and aggregated statistical metrics for the admin dashboard.
"""

from typing import Any, Dict, List, Optional, Union
import json
import os
import sqlite3

# Default DB location in workspace
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diabetescare.db",
)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a SQLite connection with foreign keys enabled and row factory set.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema if tables do not exist."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with get_connection(db_path) as conn:
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
        else:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT NOT NULL UNIQUE,
                  password_hash TEXT NOT NULL,
                  role TEXT NOT NULL CHECK(role IN ('user', 'admin')) DEFAULT 'user',
                  is_active INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS predictions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  pregnancies REAL,
                  glucose REAL,
                  blood_pressure REAL,
                  skin_thickness REAL,
                  insulin REAL,
                  bmi REAL,
                  dpf REAL,
                  age REAL,
                  probability REAL NOT NULL,
                  prediction INTEGER NOT NULL,
                  risk_band TEXT NOT NULL,
                  top_factors TEXT,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                """
            )


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

def create_user(
    name: str,
    email: str,
    password_hash: str,
    role: str = "user",
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Creates a new user record.

    Args:
        name: Full name of user.
        email: Unique email address.
        password_hash: Bcrypt-hashed password.
        role: 'user' or 'admin'.
        db_path: Path to database.

    Returns:
        The newly created user id.
    """
    clean_email = email.strip().lower()
    clean_name = name.strip()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (name, email, password_hash, role)
            VALUES (?, ?, ?, ?)
            """,
            (clean_name, clean_email, password_hash, role),
        )
        return cursor.lastrowid


def get_user_by_email(
    email: str, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetches a single user by email."""
    clean_email = email.strip().lower()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, email, password_hash, role, is_active, created_at
            FROM users
            WHERE email = ?
            """,
            (clean_email,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(
    user_id: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Fetches a single user by primary key ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, name, email, password_hash, role, is_active, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def list_all_users(
    search: str = "", db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Lists all registered users with optional search filter."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if search.strip():
            param = f"%{search.strip().lower()}%"
            cursor.execute(
                """
                SELECT id, name, email, role, is_active, created_at
                FROM users
                WHERE LOWER(name) LIKE ? OR LOWER(email) LIKE ?
                ORDER BY created_at DESC
                """,
                (param, param),
            )
        else:
            cursor.execute(
                """
                SELECT id, name, email, role, is_active, created_at
                FROM users
                ORDER BY created_at DESC
                """
            )
        return [dict(row) for row in cursor.fetchall()]


def update_user_status(
    user_id: int, is_active: int, db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Toggles user active/inactive status."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id),
        )
        return cursor.rowcount > 0


def update_user_password(
    user_id: int, new_password_hash: str, db_path: str = DEFAULT_DB_PATH
) -> bool:
    """Updates user password hash."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_password_hash, user_id),
        )
        return cursor.rowcount > 0


def delete_user(user_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a user and cascades deletion to their predictions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Prediction operations
# ---------------------------------------------------------------------------

def save_prediction(
    user_id: int,
    inputs: Dict[str, Union[int, float]],
    probability: float,
    prediction: int,
    risk_band: str,
    top_factors: Union[List[Any], Dict[str, Any], str],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Saves a risk prediction record with all input features and SHAP top factors.

    Args:
        user_id: ID of the user submitting the prediction.
        inputs: Dictionary of input clinical values.
        probability: Model prediction probability (0.0 to 1.0).
        prediction: Binary label (0 or 1).
        risk_band: 'Low', 'Moderate', or 'High'.
        top_factors: SHAP explanations (stored as serialized JSON).
        db_path: Database path.

    Returns:
        The newly inserted prediction ID.
    """
    if not isinstance(top_factors, str):
        top_factors_json = json.dumps(top_factors)
    else:
        top_factors_json = top_factors

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
              user_id, pregnancies, glucose, blood_pressure,
              skin_thickness, insulin, bmi, dpf, age,
              probability, prediction, risk_band, top_factors
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                float(inputs.get("Pregnancies", 0)),
                float(inputs.get("Glucose", 0)),
                float(inputs.get("BloodPressure", 0)),
                float(inputs.get("SkinThickness", 0)),
                float(inputs.get("Insulin", 0)),
                float(inputs.get("BMI", 0)),
                float(inputs.get("DiabetesPedigreeFunction", 0)),
                float(inputs.get("Age", 0)),
                float(probability),
                int(prediction),
                risk_band,
                top_factors_json,
            ),
        )
        return cursor.lastrowid


def get_user_predictions(
    user_id: int, db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves all past predictions for a specific user, newest first."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, pregnancies, glucose, blood_pressure,
                   skin_thickness, insulin, bmi, dpf, age,
                   probability, prediction, risk_band, top_factors, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_prediction_by_id(
    prediction_id: int, db_path: str = DEFAULT_DB_PATH
) -> Optional[Dict[str, Any]]:
    """Retrieves a single prediction by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.*, u.name as user_name, u.email as user_email
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
            """,
            (prediction_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_predictions(
    filter_band: Optional[str] = None,
    filter_user_id: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Retrieves all predictions across users with flexible filtering for admins."""
    query = """
        SELECT p.id, p.user_id, u.name as user_name, u.email as user_email,
               p.pregnancies, p.glucose, p.blood_pressure,
               p.skin_thickness, p.insulin, p.bmi, p.dpf, p.age,
               p.probability, p.prediction, p.risk_band, p.top_factors, p.created_at
        FROM predictions p
        JOIN users u ON p.user_id = u.id
        WHERE 1=1
    """
    params: List[Any] = []

    if filter_band and filter_band != "All":
        query += " AND p.risk_band = ?"
        params.append(filter_band)

    if filter_user_id:
        query += " AND p.user_id = ?"
        params.append(filter_user_id)

    if date_start:
        query += " AND date(p.created_at) >= date(?)"
        params.append(date_start)

    if date_end:
        query += " AND date(p.created_at) <= date(?)"
        params.append(date_end)

    query += " ORDER BY p.created_at DESC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_admin_dashboard_stats(
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Computes aggregated dashboard metrics for the administrator view."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Total counts
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # High risk count and percentage
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE risk_band = 'High'"
        )
        high_risk_count = cursor.fetchone()[0]
        high_risk_pct = (
            round((high_risk_count / total_predictions) * 100, 1)
            if total_predictions > 0
            else 0.0
        )

        # Risk band distribution
        cursor.execute(
            """
            SELECT risk_band, COUNT(*) as count
            FROM predictions
            GROUP BY risk_band
            """
        )
        band_distribution = {row["risk_band"]: row["count"] for row in cursor.fetchall()}

        # Predictions over time (last 30 days)
        cursor.execute(
            """
            SELECT date(created_at) as pred_date, COUNT(*) as count
            FROM predictions
            GROUP BY date(created_at)
            ORDER BY pred_date ASC
            LIMIT 30
            """
        )
        timeline = [dict(row) for row in cursor.fetchall()]

        # Average clinical input values
        cursor.execute(
            """
            SELECT
              AVG(glucose) as avg_glucose,
              AVG(blood_pressure) as avg_bp,
              AVG(bmi) as avg_bmi,
              AVG(insulin) as avg_insulin,
              AVG(age) as avg_age,
              AVG(probability) as avg_probability
            FROM predictions
            """
        )
        avg_row = cursor.fetchone()
        averages = {
            k: round(avg_row[k], 2) if avg_row and avg_row[k] is not None else 0.0
            for k in [
                "avg_glucose",
                "avg_bp",
                "avg_bmi",
                "avg_insulin",
                "avg_age",
                "avg_probability",
            ]
        }

        return {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "high_risk_count": high_risk_count,
            "high_risk_pct": high_risk_pct,
            "band_distribution": band_distribution,
            "timeline": timeline,
            "averages": averages,
        }
