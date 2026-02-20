"""
utils.py – Shared utility functions for the GA-WOA-LSTM project.

Includes:
  - Metrics calculation  (MAE, MAPE, RMSE, R²)
  - Standard chart helpers for loss curves and prediction plots
  - Inverse-scaling helper
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")


def ensure_dirs():
    """Create all required output directories if they do not already exist."""
    for d in [FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-scale helper
# ─────────────────────────────────────────────────────────────────────────────


def inverse_scale(y_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Undo MinMaxScaler normalisation for the *target* column.

    Works for both sklearn's MinMaxScaler (using data_min_ / data_max_)
    and a raw (min, max) tuple.

    Parameters
    ----------
    y_scaled : np.ndarray
        1-D array of scaled values in [0, 1].
    scaler : MinMaxScaler or tuple(float, float)
        Fitted scaler object, or a (min_val, max_val) tuple.

    Returns
    -------
    np.ndarray – values in the original price range.
    """
    if isinstance(scaler, tuple):
        min_val, max_val = scaler
    else:
        min_val = float(scaler.data_min_[-1])
        max_val = float(scaler.data_max_[-1])
    return y_scaled * (max_val - min_val) + min_val


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, MAPE, RMSE, and R² for a pair of arrays.

    Parameters
    ----------
    y_true, y_pred : np.ndarray – actual and predicted values (original scale).

    Returns
    -------
    dict with keys: MAE, MAPE, RMSE, R2
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}


def print_metrics(metrics: dict, dataset: str = "Test", model_name: str = ""):
    """Pretty-print a metrics dict."""
    prefix = f"[{model_name}] " if model_name else ""
    print(f"\n{prefix}── {dataset} Metrics ──────────────────")
    print(f"  MAE  = {metrics['MAE']:.4f}")
    print(f"  MAPE = {metrics['MAPE']:.2f}%")
    print(f"  RMSE = {metrics['RMSE']:.4f}")
    print(f"  R²   = {metrics['R2']:.4f}")


def save_metrics_csv(
    train_metrics: dict,
    test_metrics: dict,
    model_name: str,
):
    """Save train/test metrics to results/metrics/<model_name>_metrics.csv."""
    ensure_dirs()
    df = pd.DataFrame(
        {
            "Dataset": ["Train", "Test"],
            "MAE": [train_metrics["MAE"], test_metrics["MAE"]],
            "MAPE (%)": [train_metrics["MAPE"], test_metrics["MAPE"]],
            "RMSE": [train_metrics["RMSE"], test_metrics["RMSE"]],
            "R2": [train_metrics["R2"], test_metrics["R2"]],
        }
    )
    path = os.path.join(METRICS_DIR, f"{model_name}_metrics.csv")
    df.to_csv(path, index=False)
    print(f"  ✔ Metrics saved → {path}")


def save_predictions_csv(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    model_name: str,
):
    """Save train & test predictions to results/predictions/."""
    ensure_dirs()
    train_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_train_prediction.csv")
    test_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_test_prediction.csv")
    pd.DataFrame(
        {"Actual_Close_Train": y_true_train, "Predicted_Close_Train": y_pred_train}
    ).to_csv(train_path, index=False)
    pd.DataFrame({"Actual_Close": y_true_test, "Predicted_Close": y_pred_test}).to_csv(
        test_path, index=False
    )
    print(f"  ✔ Train predictions → {train_path}")
    print(f"  ✔ Test  predictions → {test_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_loss_curve(history, model_name: str):
    """
    Plot and save training vs. validation loss.

    Parameters
    ----------
    history : keras History object (has .history dict)
    model_name : str – used in the file name and title.
    """
    ensure_dirs()
    plt.figure(figsize=(16, 5))
    plt.plot(history.history["loss"], label="Train Loss", color="steelblue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="darkorange")
    plt.title(f"{model_name} – Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Loss curve saved → {path}")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset: str = "Test",
    actual_color: str = "steelblue",
    pred_color: str = "tomato",
):
    """
    Plot and save actual vs. predicted price curve.

    Parameters
    ----------
    y_true, y_pred : np.ndarray – values in original price scale.
    model_name : str – model identifier used in title and filename.
    dataset : str – "Train" or "Test" (used in title and filename).
    """
    ensure_dirs()
    plt.figure(figsize=(16, 5))
    plt.plot(y_true, label=f"Actual {dataset}", color=actual_color, linewidth=1.5)
    plt.plot(
        y_pred,
        label=f"Predicted {dataset}",
        color=pred_color,
        linewidth=1.2,
        alpha=0.85,
    )
    plt.title(f"{model_name} – {dataset} Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_{dataset.lower()}_prediction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ {dataset} prediction plot → {path}")


def plot_convergence(losses: list, model_name: str):
    """
    Plot optimisation convergence curve (for GA / WOA / GA-WOA).

    Parameters
    ----------
    losses : list of float – best score per generation.
    model_name : str.
    """
    ensure_dirs()
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker="o", color="mediumseagreen", linewidth=1.5)
    plt.title(f"{model_name} – Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best Score (lower = better)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Convergence curve → {path}")


def plot_all_models_comparison(summary_df: pd.DataFrame, metric: str = "RMSE"):
    """
    Bar chart comparing all models on a given metric.

    Parameters
    ----------
    summary_df : pd.DataFrame with columns [Model, MAE, MAPE, RMSE, R2].
    metric : str – column to plot.
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors[: len(summary_df)]
    bars = ax.bar(
        summary_df["Model"], summary_df[metric], color=colors, edgecolor="white"
    )
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title(f"Model Comparison – {metric}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✔ Comparison chart ({metric}) → {path}")
