from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class AssayModelResult:
    """
    Stores the outcome of training a single assay-specific classifier.
    """

    assay_name: str
    model: Optional[LogisticRegression]
    n_train_samples: int
    n_positive: int
    n_negative: int
    was_trained: bool
    skip_reason: Optional[str] = None


def prepare_binary_targets(
    y_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare targets for binary classification by removing missing labels.

    Parameters
    ----------
    y_series : pd.Series
        Target column for one assay.

    Returns
    -------
    valid_mask : np.ndarray
        Boolean mask for rows with non-missing labels.
    y_valid : np.ndarray
        Integer binary targets aligned to valid rows only.
    """
    valid_mask = ~y_series.isna().to_numpy()
    y_valid = y_series.loc[valid_mask].astype(int).to_numpy()
    return valid_mask, y_valid


def train_single_logreg_assay(
    X: np.ndarray,
    y_series: pd.Series,
    assay_name: str,
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    C: float = 1.0,
) -> AssayModelResult:
    """
    Train one Logistic Regression model for a single assay.

    Missing labels are removed assay-by-assay before training.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix of shape (n_samples, n_features).
    y_series : pd.Series
        Labels for one assay, may contain NaNs.
    assay_name : str
        Name of the assay column.
    random_state : int, default=42
        Random seed.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    class_weight : str or None, default="balanced"
        Class weighting strategy.
    solver : str, default="liblinear"
        Logistic Regression solver.
    C : float, default=1.0
        Inverse regularization strength.

    Returns
    -------
    AssayModelResult
        Result object containing trained model or skip reason.
    """
    valid_mask, y_valid = prepare_binary_targets(y_series)

    X_valid = X[valid_mask]

    n_train_samples = len(y_valid)
    n_positive = int((y_valid == 1).sum())
    n_negative = int((y_valid == 0).sum())

    if n_train_samples == 0:
        return AssayModelResult(
            assay_name=assay_name,
            model=None,
            n_train_samples=0,
            n_positive=0,
            n_negative=0,
            was_trained=False,
            skip_reason="No non-missing labels available.",
        )

    unique_classes = np.unique(y_valid)
    if len(unique_classes) < 2:
        return AssayModelResult(
            assay_name=assay_name,
            model=None,
            n_train_samples=n_train_samples,
            n_positive=n_positive,
            n_negative=n_negative,
            was_trained=False,
            skip_reason="Only one class present after removing missing labels.",
        )

    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        C=C,
    )

    model.fit(X_valid, y_valid)

    return AssayModelResult(
        assay_name=assay_name,
        model=model,
        n_train_samples=n_train_samples,
        n_positive=n_positive,
        n_negative=n_negative,
        was_trained=True,
        skip_reason=None,
    )


def train_logreg_multitask(
    X: np.ndarray,
    y_df: pd.DataFrame,
    assay_names: List[str],
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    C: float = 1.0,
) -> Dict[str, AssayModelResult]:
    """
    Train one Logistic Regression classifier per assay.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_df : pd.DataFrame
        DataFrame containing assay label columns.
    assay_names : list[str]
        Names of the assay columns to train on.
    random_state : int, default=42
        Random seed.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    class_weight : str or None, default="balanced"
        Class weighting strategy.
    solver : str, default="liblinear"
        Logistic Regression solver.
    C : float, default=1.0
        Inverse regularization strength.

    Returns
    -------
    dict[str, AssayModelResult]
        Dictionary of assay name -> training result.
    """
    missing_assays = [assay for assay in assay_names if assay not in y_df.columns]
    if missing_assays:
        raise ValueError(f"Missing assay columns in y_df: {missing_assays}")

    results: Dict[str, AssayModelResult] = {}

    for assay_name in assay_names:
        result = train_single_logreg_assay(
            X=X,
            y_series=y_df[assay_name],
            assay_name=assay_name,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            C=C,
        )
        results[assay_name] = result

    return results


def extract_trained_models(
    results: Dict[str, AssayModelResult],
) -> Dict[str, LogisticRegression]:
    """
    Extract only successfully trained models.

    Parameters
    ----------
    results : dict[str, AssayModelResult]
        Training results dictionary.

    Returns
    -------
    dict[str, LogisticRegression]
        Assay name -> trained model
    """
    trained_models: Dict[str, LogisticRegression] = {}

    for assay_name, result in results.items():
        if result.was_trained and result.model is not None:
            trained_models[assay_name] = result.model

    return trained_models


def summarize_training_results(
    results: Dict[str, AssayModelResult],
) -> pd.DataFrame:
    """
    Convert training results into a summary dataframe.

    Parameters
    ----------
    results : dict[str, AssayModelResult]
        Training results dictionary.

    Returns
    -------
    pd.DataFrame
        Summary table for reporting/logging.
    """
    rows = []

    for assay_name, result in results.items():
        rows.append(
            {
                "assay_name": assay_name,
                "was_trained": result.was_trained,
                "n_train_samples": result.n_train_samples,
                "n_positive": result.n_positive,
                "n_negative": result.n_negative,
                "skip_reason": result.skip_reason,
            }
        )

    return pd.DataFrame(rows)


def predict_with_trained_models(
    trained_models: Dict[str, LogisticRegression],
    X: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate class predictions and probabilities for all trained assays.

    Parameters
    ----------
    trained_models : dict[str, LogisticRegression]
        Dictionary of assay name -> trained model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    pred_df : pd.DataFrame
        Binary predictions per assay.
    proba_df : pd.DataFrame
        Positive-class probabilities per assay.
    """
    pred_dict = {}
    proba_dict = {}

    for assay_name, model in trained_models.items():
        pred_dict[assay_name] = model.predict(X)
        proba_dict[assay_name] = model.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame(pred_dict)
    proba_df = pd.DataFrame(proba_dict)

    return pred_df, proba_df