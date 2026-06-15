"""
Train an XGBoost binary classifier to predict next-day BTC price direction.

Honest baseline comparisons are always reported alongside model metrics.
Daily crypto direction prediction typically yields 52–56% accuracy; results
outside this band should be scrutinised for look-ahead leakage or overfitting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# XGBoost hyper-parameters — kept intentionally conservative to avoid overfitting
# the small crypto test set.  Do not tune these on the test set.
XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> XGBClassifier:
    """Fit an XGBClassifier on the training split.

    Args:
        X_train: Training features.
        y_train: Binary training labels.
        params: Override for XGBOOST_PARAMS.

    Returns:
        Fitted XGBClassifier.
    """
    effective_params = {**XGBOOST_PARAMS, **(params or {})}
    clf = XGBClassifier(**effective_params)
    clf.fit(X_train, y_train, verbose=False)
    logger.info("Model trained on %d samples, %d features", *X_train.shape)
    return clf


def evaluate(
    clf: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Compute classification metrics and baseline comparisons on the test set.

    Args:
        clf: Fitted classifier.
        X_test: Test features.
        y_test: True binary labels.

    Returns:
        Dictionary of metric names → values.
    """
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Baselines
    always_up_acc = y_test.mean()          # fraction of actual "up" days
    random_acc = 0.5

    metrics: Dict[str, float] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "baseline_always_up_accuracy": round(float(always_up_acc), 4),
        "baseline_random_accuracy": round(random_acc, 4),
        "n_test_samples": int(len(y_test)),
    }

    print("\n── Model metrics (test set) ────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))
    print(f"ROC-AUC : {auc:.4f}")
    print(f"\n── Baselines ───────────────────────────────────────────────")
    print(f"Always-Up accuracy : {always_up_acc:.4f}  (BTC goes up ~{always_up_acc*100:.1f}% of days in test)")
    print(f"Random classifier  : {random_acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved.")

    return metrics


def plot_feature_importance(
    clf: XGBClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """Save a bar chart of XGBoost gain-based feature importances.

    Args:
        clf: Fitted XGBClassifier.
        feature_names: Ordered list of feature names matching training columns.
        top_n: How many top features to display.
    """
    importance = pd.Series(clf.feature_importances_, index=feature_names)
    importance = importance.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("XGBoost Feature Importance (gain)")
    ax.set_xlabel("Relative importance")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved.")


def save_metrics(metrics: Dict[str, float], path: Path | str | None = None) -> None:
    """Persist metrics dict to JSON.

    Args:
        metrics: Dictionary of metric names → values.
        path: Output path; defaults to results/metrics.json.
    """
    out_path = Path(path) if path else RESULTS_DIR / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from data_loader import load_btc
    from features import build_features, chronological_split

    df = load_btc()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y)

    clf = train(X_train, y_train)
    metrics = evaluate(clf, X_test, y_test)
    plot_feature_importance(clf, list(X.columns))
    save_metrics(metrics)
