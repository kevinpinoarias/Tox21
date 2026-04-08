from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


IDENTIFIER_COLUMNS = {"mol_id", "smiles"}


def load_tox21_data(
    file_path: str | Path,
    drop_missing_smiles: bool = True,
    drop_duplicate_smiles: bool = True,
    label_handling: str = "keep",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and lightly clean the Tox21 dataset.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.
    drop_missing_smiles : bool, default=True
        Whether to drop rows where SMILES is missing.
    drop_duplicate_smiles : bool, default=True
        Whether to drop duplicate molecules based on SMILES.
    label_handling : str, default="keep"
        Strategy for label NaNs:
        - "keep": preserve missing assay labels
        - "drop_all": drop rows missing any assay label
        - "fill_zero": replace missing assay labels with 0

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe.
    label_columns : list[str]
        Assay columns used as targets.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    validate_required_columns(df)

    label_columns = get_label_columns(df)

    if drop_missing_smiles:
        df = df.dropna(subset=["smiles"])

    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""]

    if drop_duplicate_smiles:
        df = df.drop_duplicates(subset=["smiles"])

    df = apply_label_handling(df, label_columns, strategy=label_handling)

    df = df.reset_index(drop=True)

    return df, label_columns


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that required identifier columns are present."""
    missing = IDENTIFIER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def get_label_columns(df: pd.DataFrame) -> List[str]:
    """Return all assay label columns."""
    label_columns = [col for col in df.columns if col not in IDENTIFIER_COLUMNS]
    if not label_columns:
        raise ValueError("No assay label columns found.")
    return label_columns


def apply_label_handling(
    df: pd.DataFrame,
    label_columns: List[str],
    strategy: str = "keep",
) -> pd.DataFrame:
    """
    Handle missing assay labels according to the chosen strategy.
    """
    valid_strategies = {"keep", "drop_all", "fill_zero"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid label_handling='{strategy}'. "
            f"Choose from {sorted(valid_strategies)}."
        )

    if strategy == "keep":
        return df

    if strategy == "drop_all":
        return df.dropna(subset=label_columns)

    if strategy == "fill_zero":
        df = df.copy()
        df[label_columns] = df[label_columns].fillna(0)
        return df

    return df


def summarize_dataset(df: pd.DataFrame, label_columns: List[str]) -> Dict[str, object]:
    """
    Produce a high-level summary of the dataset for logging/reporting.
    """
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_labels": len(label_columns),
        "label_columns": label_columns,
        "missing_smiles": int(df["smiles"].isna().sum()),
        "missing_labels_per_assay": df[label_columns].isna().sum().to_dict(),
        "positive_labels_per_assay": df[label_columns].sum(numeric_only=True).to_dict(),
    }


def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metadata that should travel alongside predictions and error analysis.
    """
    return df[["mol_id", "smiles"]].copy()