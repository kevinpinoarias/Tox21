from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Ensure the output directory exists.

    Parameters
    ----------
    output_dir : str or Path
        Directory path.

    Returns
    -------
    Path
        Resolved directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def build_prediction_table(
    metadata_df: pd.DataFrame,
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_proba_df: pd.DataFrame,
    assay_names: List[str],
) -> pd.DataFrame:
    """
    Build a long-format prediction table for all assays.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata aligned to the feature matrix, typically containing
        columns such as mol_id and smiles.
    y_true_df : pd.DataFrame
        True labels for all assays.
    y_pred_df : pd.DataFrame
        Predicted binary labels for all assays.
    y_proba_df : pd.DataFrame
        Predicted probabilities for all assays.
    assay_names : list[str]
        Assays to include.

    Returns
    -------
    pd.DataFrame
        Long-format prediction table with one row per molecule-assay pair.
    """
    required_lengths = {len(metadata_df), len(y_true_df), len(y_pred_df), len(y_proba_df)}
    if len(required_lengths) != 1:
        raise ValueError("metadata_df, y_true_df, y_pred_df, and y_proba_df must have the same number of rows.")

    rows = []

    metadata_columns = metadata_df.columns.tolist()

    for assay_name in assay_names:
        if assay_name not in y_true_df.columns:
            raise ValueError(f"Assay '{assay_name}' not found in y_true_df.")
        if assay_name not in y_pred_df.columns:
            raise ValueError(f"Assay '{assay_name}' not found in y_pred_df.")
        if assay_name not in y_proba_df.columns:
            raise ValueError(f"Assay '{assay_name}' not found in y_proba_df.")

        assay_df = metadata_df.copy()
        assay_df["assay_name"] = assay_name
        assay_df["true_label"] = y_true_df[assay_name].values
        assay_df["pred_label"] = y_pred_df[assay_name].values
        assay_df["pred_proba"] = y_proba_df[assay_name].values

        rows.append(assay_df)

    prediction_table = pd.concat(rows, axis=0, ignore_index=True)

    prediction_table["has_true_label"] = prediction_table["true_label"].notna()
    prediction_table["true_label_clean"] = prediction_table["true_label"]
    prediction_table.loc[prediction_table["has_true_label"], "true_label_clean"] = (
        prediction_table.loc[prediction_table["has_true_label"], "true_label"].astype(int)
    )
    prediction_table["pred_label"] = prediction_table["pred_label"].astype(int)
    prediction_table["pred_proba"] = prediction_table["pred_proba"].astype(float)

    prediction_table["is_correct"] = np.where(
        prediction_table["has_true_label"],
        prediction_table["true_label_clean"] == prediction_table["pred_label"],
        np.nan,
    )

    prediction_table["error_type"] = "unknown"

    valid_mask = prediction_table["has_true_label"]

    prediction_table.loc[~valid_mask, "error_type"] = "missing_true_label"
    prediction_table.loc[
        valid_mask
        & (prediction_table["true_label_clean"] == 1)
        & (prediction_table["pred_label"] == 1),
        "error_type",
    ] = "true_positive"
    prediction_table.loc[
        valid_mask
        & (prediction_table["true_label_clean"] == 0)
        & (prediction_table["pred_label"] == 0),
        "error_type",
    ] = "true_negative"
    prediction_table.loc[
        valid_mask
        & (prediction_table["true_label_clean"] == 0)
        & (prediction_table["pred_label"] == 1),
        "error_type",
    ] = "false_positive"
    prediction_table.loc[
        valid_mask
        & (prediction_table["true_label_clean"] == 1)
        & (prediction_table["pred_label"] == 0),
        "error_type",
    ] = "false_negative"

    prediction_table["confidence"] = np.where(
        prediction_table["pred_label"] == 1,
        prediction_table["pred_proba"],
        1.0 - prediction_table["pred_proba"],
    )

    prediction_table["error_magnitude"] = np.nan
    prediction_table.loc[valid_mask, "error_magnitude"] = np.abs(
        prediction_table.loc[valid_mask, "true_label_clean"].astype(float)
        - prediction_table.loc[valid_mask, "pred_proba"]
    )

    ordered_columns = metadata_columns + [
        "assay_name",
        "true_label",
        "true_label_clean",
        "pred_label",
        "pred_proba",
        "has_true_label",
        "is_correct",
        "error_type",
        "confidence",
        "error_magnitude",
    ]

    return prediction_table[ordered_columns]


def get_misclassified_rows(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only misclassified rows with valid true labels.
    """
    required_columns = {"has_true_label", "is_correct"}
    missing = required_columns - set(prediction_table.columns)
    if missing:
        raise ValueError(f"Prediction table missing required columns: {sorted(missing)}")

    misclassified = prediction_table[
        (prediction_table["has_true_label"] == True)
        & (prediction_table["is_correct"] == False)
    ].copy()

    return misclassified.reset_index(drop=True)


def get_top_confident_errors(
    prediction_table: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the most confident misclassifications.

    These are often the most interesting errors because the model
    was wrong with high confidence.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Long-format prediction table.
    top_n : int, default=20
        Number of rows to return.

    Returns
    -------
    pd.DataFrame
        Top confident error rows.
    """
    misclassified = get_misclassified_rows(prediction_table)

    if misclassified.empty:
        return misclassified

    return (
        misclassified.sort_values(by="confidence", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def get_top_largest_probability_errors(
    prediction_table: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the rows with largest probability error magnitude.

    error_magnitude = |true_label - predicted_probability|

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Long-format prediction table.
    top_n : int, default=20
        Number of rows to return.

    Returns
    -------
    pd.DataFrame
        Top rows by error magnitude.
    """
    misclassified = get_misclassified_rows(prediction_table)

    if misclassified.empty:
        return misclassified

    return (
        misclassified.sort_values(by="error_magnitude", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def summarize_errors_by_assay(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize classification outcomes and error counts per assay.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Long-format prediction table.

    Returns
    -------
    pd.DataFrame
        Assay-level error summary.
    """
    valid_df = prediction_table[prediction_table["has_true_label"] == True].copy()

    if valid_df.empty:
        return pd.DataFrame()

    summary_rows = []

    for assay_name, assay_df in valid_df.groupby("assay_name"):
        n_total = len(assay_df)
        n_correct = int((assay_df["is_correct"] == True).sum())
        n_incorrect = int((assay_df["is_correct"] == False).sum())
        n_fp = int((assay_df["error_type"] == "false_positive").sum())
        n_fn = int((assay_df["error_type"] == "false_negative").sum())
        n_tp = int((assay_df["error_type"] == "true_positive").sum())
        n_tn = int((assay_df["error_type"] == "true_negative").sum())

        error_rate = n_incorrect / n_total if n_total > 0 else np.nan
        fp_rate_within_errors = n_fp / n_incorrect if n_incorrect > 0 else np.nan
        fn_rate_within_errors = n_fn / n_incorrect if n_incorrect > 0 else np.nan

        summary_rows.append(
            {
                "assay_name": assay_name,
                "n_total": n_total,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "n_true_positive": n_tp,
                "n_true_negative": n_tn,
                "n_false_positive": n_fp,
                "n_false_negative": n_fn,
                "error_rate": error_rate,
                "fp_rate_within_errors": fp_rate_within_errors,
                "fn_rate_within_errors": fn_rate_within_errors,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(by="error_rate", ascending=False).reset_index(drop=True)


def summarize_error_types(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize overall counts of error types across all assays.
    """
    summary = (
        prediction_table["error_type"]
        .value_counts(dropna=False)
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    total = summary["count"].sum()
    summary["fraction"] = summary["count"] / total if total > 0 else np.nan

    return summary


def find_hardest_assays(
    error_summary_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the assays with the highest error rate.
    """
    if error_summary_df.empty:
        return error_summary_df

    return (
        error_summary_df.sort_values(by="error_rate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def find_easiest_assays(
    error_summary_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the assays with the lowest error rate.
    """
    if error_summary_df.empty:
        return error_summary_df

    return (
        error_summary_df.sort_values(by="error_rate", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def find_false_positives(
    prediction_table: pd.DataFrame,
    assay_name: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_by_confidence: bool = True,
) -> pd.DataFrame:
    """
    Return false positive rows, optionally filtered by assay.
    """
    df = prediction_table[prediction_table["error_type"] == "false_positive"].copy()

    if assay_name is not None:
        df = df[df["assay_name"] == assay_name].copy()

    if sort_by_confidence and not df.empty:
        df = df.sort_values(by="confidence", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    return df.reset_index(drop=True)


def find_false_negatives(
    prediction_table: pd.DataFrame,
    assay_name: Optional[str] = None,
    top_n: Optional[int] = None,
    sort_by_confidence: bool = True,
) -> pd.DataFrame:
    """
    Return false negative rows, optionally filtered by assay.
    """
    df = prediction_table[prediction_table["error_type"] == "false_negative"].copy()

    if assay_name is not None:
        df = df[df["assay_name"] == assay_name].copy()

    if sort_by_confidence and not df.empty:
        df = df.sort_values(by="confidence", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    return df.reset_index(drop=True)


def identify_rare_positive_assays(
    per_assay_metrics_df: pd.DataFrame,
    positive_rate_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Identify assays with a low positive-class rate.

    Parameters
    ----------
    per_assay_metrics_df : pd.DataFrame
        Per-assay metrics dataframe containing n_positive and n_negative.
    positive_rate_threshold : float, default=0.1
        Threshold below which an assay is considered rare-positive.

    Returns
    -------
    pd.DataFrame
        Assays with positive rate below threshold.
    """
    required_columns = {"assay_name", "n_positive", "n_negative"}
    missing = required_columns - set(per_assay_metrics_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = per_assay_metrics_df.copy()
    df["positive_rate"] = df["n_positive"] / (df["n_positive"] + df["n_negative"])
    df = df[df["positive_rate"] < positive_rate_threshold].copy()

    return df.sort_values(by="positive_rate", ascending=True).reset_index(drop=True)


def save_error_analysis_tables(
    prediction_table: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 20,
) -> Dict[str, Path]:
    """
    Save standard error-analysis tables to disk.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Long-format prediction table.
    output_dir : str or Path
        Directory where CSV files will be saved.
    top_n : int, default=20
        Number of top error rows to save for ranked tables.

    Returns
    -------
    dict[str, Path]
        Mapping of table name to saved file path.
    """
    output_dir = ensure_output_dir(output_dir)

    files = {}

    prediction_path = output_dir / "prediction_table.csv"
    prediction_table.to_csv(prediction_path, index=False)
    files["prediction_table"] = prediction_path

    misclassified_df = get_misclassified_rows(prediction_table)
    misclassified_path = output_dir / "misclassified_rows.csv"
    misclassified_df.to_csv(misclassified_path, index=False)
    files["misclassified_rows"] = misclassified_path

    confident_errors_df = get_top_confident_errors(prediction_table, top_n=top_n)
    confident_errors_path = output_dir / "top_confident_errors.csv"
    confident_errors_df.to_csv(confident_errors_path, index=False)
    files["top_confident_errors"] = confident_errors_path

    largest_errors_df = get_top_largest_probability_errors(prediction_table, top_n=top_n)
    largest_errors_path = output_dir / "top_probability_errors.csv"
    largest_errors_df.to_csv(largest_errors_path, index=False)
    files["top_probability_errors"] = largest_errors_path

    assay_summary_df = summarize_errors_by_assay(prediction_table)
    assay_summary_path = output_dir / "error_summary_by_assay.csv"
    assay_summary_df.to_csv(assay_summary_path, index=False)
    files["error_summary_by_assay"] = assay_summary_path

    error_types_df = summarize_error_types(prediction_table)
    error_types_path = output_dir / "error_type_summary.csv"
    error_types_df.to_csv(error_types_path, index=False)
    files["error_type_summary"] = error_types_path

    return files