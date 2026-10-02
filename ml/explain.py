"""Model explainability module using SHAP for DiabetesCare AI.

Provides local patient-level attribution, plain-English summary translations,
top-contributing feature extraction, and global population-level visual insights.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.preprocess import FEATURE_COLUMNS

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
EXPLAINER_PATH = os.path.join(ARTIFACTS_DIR, "explainer.joblib")


def get_tree_explainer(model: Any) -> Any:
    """Loads a cached SHAP TreeExplainer or fits a new one on the provided model.

    Args:
        model: Trained scikit-learn tree-based model (RandomForestClassifier).

    Returns:
        shap.TreeExplainer instance.
    """
    import shap

    if os.path.exists(EXPLAINER_PATH):
        try:
            return joblib.load(EXPLAINER_PATH)
        except Exception:
            pass

    explainer = shap.TreeExplainer(model)
    return explainer


def extract_positive_class_shap(shap_output: Any) -> np.ndarray:
    """Robustly extracts SHAP values corresponding to the positive class (Outcome = 1).

    Accounts for differences between SHAP library versions:
    - Version producing a list of 2D arrays: [class_0, class_1]
    - Version producing a 3D array: (n_samples, n_features, n_classes)
    - Version producing an Explanation object

    Args:
        shap_output: Raw output from explainer.shap_values() or explainer().

    Returns:
        1D or 2D numpy array of SHAP values for class 1.
    """
    if hasattr(shap_output, "values"):
        vals = shap_output.values
        if len(vals.shape) == 3 and vals.shape[2] == 2:
            return vals[:, :, 1]
        return vals

    if isinstance(shap_output, list):
        if len(shap_output) >= 2:
            return np.array(shap_output[1])
        return np.array(shap_output[0])

    arr = np.array(shap_output)
    if len(arr.shape) == 3 and arr.shape[2] == 2:
        return arr[:, :, 1]
    return arr


def create_local_explanation_plot(
    features: List[str],
    feature_values: List[float],
    shap_vals: List[float],
    probability: float,
) -> plt.Figure:
    """Generates a clean, professional horizontal bar chart showing local SHAP attribution.

    Positive SHAP values (pushing risk UP) are colored in Coral Red.
    Negative SHAP values (reducing risk / protective) are colored in Teal Green.

    Args:
        features: Feature names.
        feature_values: Patient's clinical values.
        shap_vals: SHAP contribution values.
        probability: Patient's predicted diabetes probability.

    Returns:
        matplotlib Figure object.
    """
    # Create combined dataframe and sort by absolute contribution
    plot_df = pd.DataFrame({
        "feature": features,
        "value": feature_values,
        "shap": shap_vals,
    })
    plot_df["abs_shap"] = plot_df["shap"].abs()
    plot_df = plot_df.sort_values("abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    colors = ["#ef4444" if val > 0 else "#10b981" for val in plot_df["shap"]]
    bars = ax.barh(
        range(len(plot_df)),
        plot_df["shap"],
        color=colors,
        height=0.62,
        edgecolor="none",
        alpha=0.9,
    )

    # Format y-tick labels with feature name and patient's value
    y_labels = [
        f"{row['feature']}  ({row['value']:.1f})"
        if isinstance(row["value"], float) and not row["value"].is_integer()
        else f"{row['feature']}  ({int(row['value'])})"
        for _, row in plot_df.iterrows()
    ]
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="500", color="#1e293b")

    # Add reference line at zero
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8)

    # Annotate value labels on bars
    for bar in bars:
        width = bar.get_width()
        x_pos = width + (0.005 if width >= 0 else -0.005)
        ha = "left" if width >= 0 else "right"
        sign = "+" if width > 0 else ""
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{width:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="600",
            color="#334155",
        )

    # Clean borders and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#94a3b8")

    ax.set_xlabel(
        "SHAP Value (Impact on Model Risk Score: Red = Increases Risk, Green = Lowers Risk)",
        fontsize=9.5,
        color="#475569",
        labelpad=10,
    )
    ax.set_title(
        f"Individual Feature Attribution Breakdown (Predicted Risk: {probability:.1%})",
        fontsize=11.5,
        fontweight="bold",
        color="#0f172a",
        pad=14,
    )

    plt.tight_layout()
    return fig


def explain_prediction(
    model: Any,
    processed_df: pd.DataFrame,
    probability: float,
    explainer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generates local SHAP explanation, top factors list, narrative, and plot.

    Args:
        model: Trained RandomForest model.
        processed_df: 1-row DataFrame containing preprocessed inputs.
        probability: Patient's predicted diabetes probability.
        explainer: Optional cached TreeExplainer.

    Returns:
        Dictionary containing top_factors, summary_narrative, shap_dict, and figure.
    """
    import shap

    if explainer is None:
        explainer = get_tree_explainer(model)

    raw_shap = explainer.shap_values(processed_df)
    shap_positive = extract_positive_class_shap(raw_shap)

    # Ensure 1D array of length n_features
    if len(shap_positive.shape) > 1:
        shap_row = shap_positive[0]
    else:
        shap_row = shap_positive

    features = list(processed_df.columns)
    values = [float(processed_df.iloc[0][f]) for f in features]
    shap_list = [float(val) for val in shap_row]

    # Create top factors list
    factors = []
    for f, v, s in zip(features, values, shap_list):
        direction = "increased" if s > 0 else "decreased"
        factors.append({
            "feature": f,
            "value": round(v, 2),
            "shap_value": round(s, 4),
            "abs_impact": round(abs(s), 4),
            "direction": direction,
        })

    # Sort factors by impact magnitude
    factors_sorted = sorted(factors, key=lambda x: x["abs_impact"], reverse=True)

    # Build plain-English narrative
    top_increase = [f for f in factors_sorted if f["direction"] == "increased"]
    top_decrease = [f for f in factors_sorted if f["direction"] == "decreased"]

    sentences: List[str] = []
    if top_increase:
        top_inc = top_increase[0]
        sentences.append(
            f"Your **{top_inc['feature']}** ({top_inc['value']}) increased your estimated risk the most."
        )
        if len(top_increase) > 1:
            second_inc = top_increase[1]
            sentences.append(
                f"Elevated **{second_inc['feature']}** ({second_inc['value']}) also contributed positively to your risk score."
            )

    if top_decrease:
        top_dec = top_decrease[0]
        sentences.append(
            f"Conversely, your **{top_dec['feature']}** ({top_dec['value']}) served as the strongest protective factor lowering your score."
        )

    narrative = " ".join(sentences)

    # Generate explanation figure
    fig = create_local_explanation_plot(
        features=features,
        feature_values=values,
        shap_vals=shap_list,
        probability=probability,
    )

    return {
        "top_factors": factors_sorted,
        "summary_narrative": narrative,
        "shap_dict": {f: s for f, s in zip(features, shap_list)},
        "figure": fig,
    }
