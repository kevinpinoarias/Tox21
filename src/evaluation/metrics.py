from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _get_valid_assay_rows(y_true: pd.Series) -> np.ndarray:
    """
    Return a boolean mask for rows where the true assay label is not missing.
    """
    return ~y_true.isna().to_numpy()


def evaluate_single_assay(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
    assay_name: str,
) -> Dict[str, object]:
    """
    Evaluate one assay using valid (non-missing) rows only.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels for one assay, may contain NaNs.
    y_pred : pd.Series
        Predicted binary labels for the same assay.
    y_proba : pd.Series
        Predicted positive-class probabilities for the same assay.
    assay_name : str
        Name of the assay.

    Returns
    -------
    dict
        Dictionary of assay-level evaluation metrics.
    """
    valid_mask = _get_valid_assay_rows(y_true)

    y_true_valid = y_true.loc[valid_mask].astype(int)
    y_pred_valid = y_pred.loc[valid_mask].astype(int)
    y_proba_valid = y_proba.loc[valid_mask].astype(float)

    n_samples = len(y_true_valid)
    n_positive = int((y_true_valid == 1).sum())
    n_negative = int((y_true_valid == 0).sum())

    if n_samples == 0:
        return {
            "assay_name": assay_name,
            "n_samples": 0,
            "n_positive": 0,
            "n_negative": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "evaluation_status": "skipped_no_valid_labels",
        }

    unique_classes = y_true_valid.nunique()

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision = precision_score(y_true_valid, y_pred_valid, zero_division=0)
    recall = recall_score(y_true_valid, y_pred_valid, zero_division=0)
    f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)

    if unique_classes < 2:
        roc_auc = np.nan
        pr_auc = np.nan
        status = "evaluated_single_class_only"
    else:
        roc_auc = roc_auc_score(y_true_valid, y_proba_valid)
        pr_auc = average_precision_score(y_true_valid, y_proba_valid)
        status = "evaluated"

    return {
        "assay_name": assay_name,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "evaluation_status": status,
    }


def evaluate_multitask_predictions(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
) -> pd.DataFrame:
    """
    Evaluate predictions assay-by-assay.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        True assay labels, may contain NaNs.
    y_pred_df : pd.DataFrame
        Predicted binary labels.
    y_proba_df : pd.DataFrame
        Predicted probabilities.
    assay_names : list[str]
        Assays to evaluate.

    Returns
    -------
    pd.DataFrame
        Per-assay metrics table.
    """
    missing_true = [a for a in assay_names if a not in y_true_df.columns]
    missing_pred = [a for a in assay_names if a not in y_pred_df.columns]
    missing_proba = [a for a in assay_names if a not in y_proba_df.columns]

    if missing_true:
        raise ValueError(f"Missing assays in y_true_df: {missing_true}")
    if missing_pred:
        raise ValueError(f"Missing assays in y_pred_df: {missing_pred}")
    if missing_proba:
        raise ValueError(f"Missing assays in y_proba_df: {missing_proba}")

    results = []

    for assay_name in assay_names:
        assay_result = evaluate_single_assay(
            y_true=y_true_df[assay_name],
            y_pred=y_pred_df[assay_name],
            y_proba=y_proba_df[assay_name],
            assay_name=assay_name,
        )
        results.append(assay_result)

    return pd.DataFrame(results)


def compute_macro_metrics(
    per_assay_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute macro-average metrics across assays.

    Macro-average = average of assay-level metrics.
    Each assay contributes equally.

    Parameters
    ----------
    per_assay_df : pd.DataFrame
        Output of evaluate_multitask_predictions().

    Returns
    -------
    dict
        Macro-averaged metrics.
    """
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    return {
        f"macro_{metric}": float(per_assay_df[metric].mean(skipna=True))
        for metric in metric_columns
    }


def compute_micro_metrics(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
) -> Dict[str, float]:
    """
    Compute micro-average metrics across all assays and all valid rows.

    Micro-average = flatten all assay predictions into one big pool,
    ignoring missing true labels.

    Parameters
    ----------
    y_true_df : pd.DataFrame
        True labels.
    y_pred_df : pd.DataFrame
        Predicted labels.
    y_proba_df : pd.DataFrame
        Predicted probabilities.
    assay_names : list[str]
        Assays to include.

    Returns
    -------
    dict
        Micro-averaged metrics.
    """
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for assay_name in assay_names:
        y_true = y_true_df[assay_name]
        valid_mask = _get_valid_assay_rows(y_true)

        y_true_all.append(y_true.loc[valid_mask].astype(int).to_numpy())
        y_pred_all.append(y_pred_df.loc[valid_mask, assay_name].astype(int).to_numpy())
        y_proba_all.append(y_proba_df.loc[valid_mask, assay_name].astype(float).to_numpy())

    if not y_true_all:
        return {
            "micro_accuracy": np.nan,
            "micro_precision": np.nan,
            "micro_recall": np.nan,
            "micro_f1": np.nan,
            "micro_roc_auc": np.nan,
            "micro_pr_auc": np.nan,
        }

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)
    y_proba_concat = np.concatenate(y_proba_all)

    if len(y_true_concat) == 0:
        return {
            "micro_accuracy": np.nan,
            "micro_precision": np.nan,
            "micro_recall": np.nan,
            "micro_f1": np.nan,
            "micro_roc_auc": np.nan,
            "micro_pr_auc": np.nan,
        }

    accuracy = accuracy_score(y_true_concat, y_pred_concat)
    precision = precision_score(y_true_concat, y_pred_concat, zero_division=0)
    recall = recall_score(y_true_concat, y_pred_concat, zero_division=0)
    f1 = f1_score(y_true_concat, y_pred_concat, zero_division=0)

    if np.unique(y_true_concat).size < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true_concat, y_proba_concat)
        pr_auc = average_precision_score(y_true_concat, y_proba_concat)

    return {
        "micro_accuracy": float(accuracy),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "micro_roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "micro_pr_auc": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
    }


def build_overall_metrics_summary(
    per_assay_df: pd.DataFrame,
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
) -> pd.DataFrame:
    """
    Combine macro and micro metrics into one compact summary dataframe.

    Returns
    -------
    pd.DataFrame
        One-row summary dataframe.
    """
    macro_metrics = compute_macro_metrics(per_assay_df)
    micro_metrics = compute_micro_metrics(
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        y_proba_df=y_proba_df,
        assay_names=assay_names,
    )

    combined = {**macro_metrics, **micro_metrics}
    return pd.DataFrame([combined])


def rank_assays_by_metric(
    per_assay_df: pd.DataFrame,
    metric: str = "roc_auc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank assays by a chosen metric.

    Parameters
    ----------
    per_assay_df : pd.DataFrame
        Per-assay metrics dataframe.
    metric : str, default="roc_auc"
        Metric to sort by.
    ascending : bool, default=False
        Sort order.

    Returns
    -------
    pd.DataFrame
        Sorted dataframe.
    """
    if metric not in per_assay_df.columns:
        raise ValueError(f"Metric '{metric}' not found in per_assay_df.")

    return per_assay_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)


def get_best_and_worst_assays(
    per_assay_df: pd.DataFrame,
    metric: str = "roc_auc",
) -> Dict[str, Optional[Dict[str, object]]]:
    """
    Return the best and worst assay according to a chosen metric.

    Parameters
    ----------
    per_assay_df : pd.DataFrame
        Per-assay metrics dataframe.
    metric : str, default="roc_auc"
        Metric used for comparison.

    Returns
    -------
    dict
        Dictionary with 'best' and 'worst' assay summaries.
    """
    if metric not in per_assay_df.columns:
        raise ValueError(f"Metric '{metric}' not found in per_assay_df.")

    valid_df = per_assay_df.dropna(subset=[metric]).copy()

    if valid_df.empty:
        return {"best": None, "worst": None}

    best_row = valid_df.sort_values(metric, ascending=False).iloc[0].to_dict()
    worst_row = valid_df.sort_values(metric, ascending=True).iloc[0].to_dict()

    return {"best": best_row, "worst": worst_row}