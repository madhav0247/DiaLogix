"""Training pipeline for DiabetesCare AI.

Trains a RandomForestClassifier on the Pima Indians Diabetes dataset,
optimizing for recall via GridSearchCV, saves evaluation metrics and model artifacts,
and generates global SHAP explainability plots.
"""

from typing import Any, Dict, List, Tuple
import json
import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ZERO_AS_MISSING_COLUMNS,
    compute_training_medians,
    impute_missing_values,
    replace_invalid_zeros,
    save_medians,
)

DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def extract_positive_shap(shap_values: Any) -> np.ndarray:
    """Extracts SHAP values for the positive class (diabetic, label 1).

    Handles variations across SHAP versions (list of arrays, 3D array, or 2D array).

    Args:
        shap_values: Output from shap.TreeExplainer.shap_values() or explainer().

    Returns:
        2D numpy array of shape (n_samples, n_features) for class 1.
    """
    if hasattr(shap_values, "values"):
        # SHAP Explanation object
        vals = shap_values.values
        if len(vals.shape) == 3 and vals.shape[2] == 2:
            return vals[:, :, 1]
        return vals

    if isinstance(shap_values, list):
        # List of [class_0_vals, class_1_vals]
        if len(shap_values) >= 2:
            return np.array(shap_values[1])
        return np.array(shap_values[0])

    arr = np.array(shap_values)
    if len(arr.shape) == 3 and arr.shape[2] == 2:
        return arr[:, :, 1]
    return arr


def train_pipeline(
    data_path: str = DATA_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Executes the full end-to-end model training, validation, and artifact export.

    Args:
        data_path: Path to raw diabetes.csv.
        artifacts_dir: Destination directory for model and metrics artifacts.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary of final test evaluation metrics.
    """
    print("=" * 65)
    print("  DiabetesCare AI: Model Training & Artifact Generation Pipeline")
    print("=" * 65)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load Data
    print(f"[*] Step 1: Loading raw data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Loaded {len(df)} records with columns: {list(df.columns)}")

    # 2. Convert invalid zeros to NaN in physiological features
    print("[*] Step 2: Replacing invalid physiological 0 values with NaN...")
    X_raw = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    X_clean = replace_invalid_zeros(X_raw, ZERO_AS_MISSING_COLUMNS)

    # 3. Stratified Train/Test Split (80/20)
    print("[*] Step 3: Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    print(f"    Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"    Train diabetic prevalence: {y_train.mean():.1%}")
    print(f"    Test diabetic prevalence:  {y_test.mean():.1%}")

    # 4. Impute strictly on training medians (prevent leakage)
    print("[*] Step 4: Computing training medians and imputing missing values...")
    train_medians = compute_training_medians(X_train, ZERO_AS_MISSING_COLUMNS)
    print(f"    Computed medians: {train_medians}")

    X_train_imp = impute_missing_values(X_train, train_medians)
    X_test_imp = impute_missing_values(X_test, train_medians)

    # Save medians in JSON and joblib
    save_medians(train_medians, os.path.join(artifacts_dir, "medians.json"))
    save_medians(train_medians, os.path.join(artifacts_dir, "imputer.joblib"))
    print("    [+] Saved medians to artifacts/medians.json and imputer.joblib")

    # 5. Train RandomForestClassifier with GridSearchCV
    print("[*] Step 5: Hyperparameter tuning via GridSearchCV (5-Fold, Recall-focused)...")
    base_rf = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",  # Critical for healthcare to penalize false negatives
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="recall",  # Prioritizing clinical recall
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_imp, y_train)
    best_model: RandomForestClassifier = grid_search.best_estimator_

    print(f"    [+] Best CV Recall: {grid_search.best_score_:.4f}")
    print(f"    [+] Best Hyperparameters: {grid_search.best_params_}")

    # 6. Evaluation on Held-Out Test Set
    print("[*] Step 6: Evaluating model on held-out test set...")
    y_pred = best_model.predict(X_test_imp)
    y_prob = best_model.predict_proba(X_test_imp)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data = {
        "fpr": [round(float(val), 4) for val in fpr],
        "tpr": [round(float(val), 4) for val in tpr],
    }

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "best_params": grid_search.best_params_,
        "feature_names": FEATURE_COLUMNS,
        "test_sample_count": len(y_test),
        "roc_curve": roc_data,
    }

    print("\n" + "-" * 40)
    print("  TEST EVALUATION METRICS:")
    print("-" * 40)
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  <-- PRIORITY METRIC")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("-" * 40 + "\n")

    # Save model and metrics
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(best_model, model_path)
    print(f"[+] Saved model to {model_path}")

    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics to {metrics_path}")

    # 7. SHAP Explainability & Global Plots
    print("[*] Step 7: Fitting SHAP TreeExplainer and generating global plots...")
    try:
        import shap

        explainer = shap.TreeExplainer(best_model)
        explainer_path = os.path.join(artifacts_dir, "explainer.joblib")
        joblib.dump(explainer, explainer_path)
        print(f"[+] Saved TreeExplainer to {explainer_path}")

        # Compute SHAP on test data
        shap_vals_raw = explainer.shap_values(X_test_imp)
        shap_positive = extract_positive_shap(shap_vals_raw)

        # Generate and save global SHAP summary beeswarm/bar plot
        fig, ax = plt.subplots(figsize=(9, 5.5))
        shap.summary_plot(
            shap_positive,
            X_test_imp,
            feature_names=FEATURE_COLUMNS,
            show=False,
        )
        plt.title(
            "Global Feature Importance: Impact on Diabetes Risk (SHAP)",
            fontsize=12,
            pad=14,
            fontweight="bold",
        )
        plt.tight_layout()
        global_shap_path = os.path.join(artifacts_dir, "global_shap.png")
        plt.savefig(global_shap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Saved global SHAP plot to {global_shap_path}")

        # Also save reference background data for SHAP water/force plots
        background_path = os.path.join(artifacts_dir, "shap_background.joblib")
        joblib.dump(X_train_imp.iloc[:100], background_path)

    except Exception as e:
        print(f"[!] Warning during SHAP generation: {e}")

    print("=" * 65)
    print("  Pipeline Completed Successfully!")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    train_pipeline()
