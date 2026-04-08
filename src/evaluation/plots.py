from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Ensure the plot output directory exists.

    Parameters
    ----------
    output_dir : str or Path
        Directory path for saving plots.

    Returns
    -------
    Path
        Resolved output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _get_valid_assay_rows(y_true: pd.Series) -> np.ndarray:
    """
    Return a boolean mask for non-missing labels.
    """
    return ~y_true.isna().to_numpy()


def plot_roc_curve_for_assay(
    y_true: pd.Series,
    y_proba: pd.Series,
    assay_name: str,
    output_dir: str | Path,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot and save ROC curve for a single assay.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels, may contain NaNs.
    y_proba : pd.Series
        Predicted probabilities for positive class.
    assay_name : str
        Assay name.
    output_dir : str or Path
        Directory to save plot.
    filename : str, optional
        Custom filename. If None, a default one is created.

    Returns
    -------
    Path or None
        Saved plot path, or None if plotting was skipped.
    """
    valid_mask = _get_valid_assay_rows(y_true)
    y_true_valid = y_true.loc[valid_mask].astype(int)
    y_proba_valid = y_proba.loc[valid_mask].astype(float)

    if len(y_true_valid) == 0 or y_true_valid.nunique() < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true_valid, y_proba_valid)

    output_dir = ensure_output_dir(output_dir)

    if filename is None:
        filename = f"roc_curve_{assay_name}.png"

    output_path = output_dir / filename

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{assay_name}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {assay_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_pr_curve_for_assay(
    y_true: pd.Series,
    y_proba: pd.Series,
    assay_name: str,
    output_dir: str | Path,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot and save Precision-Recall curve for a single assay.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels, may contain NaNs.
    y_proba : pd.Series
        Predicted probabilities for positive class.
    assay_name : str
        Assay name.
    output_dir : str or Path
        Directory to save plot.
    filename : str, optional
        Custom filename. If None, a default one is created.

    Returns
    -------
    Path or None
        Saved plot path, or None if plotting was skipped.
    """
    valid_mask = _get_valid_assay_rows(y_true)
    y_true_valid = y_true.loc[valid_mask].astype(int)
    y_proba_valid = y_proba.loc[valid_mask].astype(float)

    if len(y_true_valid) == 0 or y_true_valid.nunique() < 2:
        return None

    precision, recall, _ = precision_recall_curve(y_true_valid, y_proba_valid)

    output_dir = ensure_output_dir(output_dir)

    if filename is None:
        filename = f"pr_curve_{assay_name}.png"

    output_path = output_dir / filename

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{assay_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {assay_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_all_roc_curves(
    y_true_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
    output_dir: str | Path,
) -> List[Path]:
    """
    Generate ROC curve plots for all assays.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        True assay labels.
    y_proba_df : pd.DataFrame
        Predicted probabilities.
    assay_names : list[str]
        Assays to plot.
    output_dir : str or Path
        Directory to save plots.

    Returns
    -------
    list[Path]
        List of saved plot paths.
    """
    saved_paths: List[Path] = []

    for assay_name in assay_names:
        if assay_name not in y_true_df.columns or assay_name not in y_proba_df.columns:
            continue

        path = plot_roc_curve_for_assay(
            y_true=y_true_df[assay_name],
            y_proba=y_proba_df[assay_name],
            assay_name=assay_name,
            output_dir=output_dir,
        )
        if path is not None:
            saved_paths.append(path)

    return saved_paths


def plot_all_pr_curves(
    y_true_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
    output_dir: str | Path,
) -> List[Path]:
    """
    Generate Precision-Recall curve plots for all assays.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        True assay labels.
    y_proba_df : pd.DataFrame
        Predicted probabilities.
    assay_names : list[str]
        Assays to plot.
    output_dir : str or Path
        Directory to save plots.

    Returns
    -------
    list[Path]
        List of saved plot paths.
    """
    saved_paths: List[Path] = []

    for assay_name in assay_names:
        if assay_name not in y_true_df.columns or assay_name not in y_proba_df.columns:
            continue

        path = plot_pr_curve_for_assay(
            y_true=y_true_df[assay_name],
            y_proba=y_proba_df[assay_name],
            assay_name=assay_name,
            output_dir=output_dir,
        )
        if path is not None:
            saved_paths.append(path)

    return saved_paths


def plot_metric_bar_chart(
    per_assay_df: pd.DataFrame,
    metric: str,
    output_dir: str | Path,
    filename: Optional[str] = None,
    sort_descending: bool = True,
) -> Path:
    """
    Plot a bar chart of one metric across assays.

    Parameters
    ----------
    per_assay_df : pd.DataFrame
        Per-assay metrics dataframe.
    metric : str
        Metric column to plot.
    output_dir : str or Path
        Directory to save plot.
    filename : str, optional
        Custom filename.
    sort_descending : bool, default=True
        Whether to sort assays by metric value.

    Returns
    -------
    Path
        Saved plot path.
    """
    if metric not in per_assay_df.columns:
        raise ValueError(f"Metric '{metric}' not found in per_assay_df.")

    plot_df = per_assay_df.dropna(subset=[metric]).copy()

    if sort_descending:
        plot_df = plot_df.sort_values(by=metric, ascending=False)
    else:
        plot_df = plot_df.sort_values(by=metric, ascending=True)

    output_dir = ensure_output_dir(output_dir)

    if filename is None:
        filename = f"bar_chart_{metric}.png"

    output_path = output_dir / filename

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["assay_name"], plot_df[metric])
    plt.xlabel("Assay")
    plt.ylabel(metric)
    plt.title(f"{metric} by Assay")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_class_balance(
    per_assay_df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "class_balance_per_assay.png",
) -> Path:
    """
    Plot positive and negative label counts per assay.

    Parameters
    ----------
    per_assay_df : pd.DataFrame
        Per-assay metrics dataframe containing n_positive and n_negative.
    output_dir : str or Path
        Directory to save plot.
    filename : str, default='class_balance_per_assay.png'
        Output file name.

    Returns
    -------
    Path
        Saved plot path.
    """
    required_columns = {"assay_name", "n_positive", "n_negative"}
    missing = required_columns - set(per_assay_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for class balance plot: {sorted(missing)}")

    output_dir = ensure_output_dir(output_dir)
    output_path = output_dir / filename

    plot_df = per_assay_df.copy()
    x = np.arange(len(plot_df))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, plot_df["n_positive"], width=width, label="Positive")
    plt.bar(x + width / 2, plot_df["n_negative"], width=width, label="Negative")
    plt.xlabel("Assay")
    plt.ylabel("Count")
    plt.title("Class Balance by Assay")
    plt.xticks(x, plot_df["assay_name"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def generate_standard_evaluation_plots(
    y_true_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    per_assay_df: pd.DataFrame,
    assay_names: List[str],
    output_dir: str | Path,
) -> dict:
    """
    Generate a standard set of evaluation plots for the project.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        True labels.
    y_proba_df : pd.DataFrame
        Predicted probabilities.
    per_assay_df : pd.DataFrame
        Per-assay metrics dataframe.
    assay_names : list[str]
        Assays to include.
    output_dir : str or Path
        Base output directory.

    Returns
    -------
    dict
        Dictionary of saved plot paths.
    """
    output_dir = ensure_output_dir(output_dir)

    roc_dir = output_dir / "roc_curves"
    pr_dir = output_dir / "pr_curves"
    metric_dir = output_dir / "metric_bars"
    balance_dir = output_dir / "class_balance"

    roc_paths = plot_all_roc_curves(
        y_true_df=y_true_df,
        y_proba_df=y_proba_df,
        assay_names=assay_names,
        output_dir=roc_dir,
    )

    pr_paths = plot_all_pr_curves(
        y_true_df=y_true_df,
        y_proba_df=y_proba_df,
        assay_names=assay_names,
        output_dir=pr_dir,
    )

    roc_bar = plot_metric_bar_chart(
        per_assay_df=per_assay_df,
        metric="roc_auc",
        output_dir=metric_dir,
        filename="roc_auc_by_assay.png",
    )

    f1_bar = plot_metric_bar_chart(
        per_assay_df=per_assay_df,
        metric="f1",
        output_dir=metric_dir,
        filename="f1_by_assay.png",
    )

    pr_auc_bar = plot_metric_bar_chart(
        per_assay_df=per_assay_df,
        metric="pr_auc",
        output_dir=metric_dir,
        filename="pr_auc_by_assay.png",
    )

    class_balance_plot = plot_class_balance(
        per_assay_df=per_assay_df,
        output_dir=balance_dir,
    )

    return {
        "roc_curve_paths": roc_paths,
        "pr_curve_paths": pr_paths,
        "roc_auc_bar_chart": roc_bar,
        "f1_bar_chart": f1_bar,
        "pr_auc_bar_chart": pr_auc_bar,
        "class_balance_plot": class_balance_plot,
    }