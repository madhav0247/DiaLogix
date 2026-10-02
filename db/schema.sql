-- Database schema for DiabetesCare AI

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
  top_factors TEXT,              -- JSON string of top SHAP contributors
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast user prediction lookups
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
