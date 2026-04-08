from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class FingerprintResult:
    """
    Container for featurization outputs.

    Attributes
    ----------
    X : np.ndarray
        Fingerprint matrix of shape (n_valid_molecules, n_bits).
    valid_indices : list[int]
        Original row indices from the input dataframe/list that produced valid molecules.
    invalid_indices : list[int]
        Original row indices that failed SMILES parsing.
    valid_smiles : list[str]
        SMILES strings successfully converted into fingerprints.
    """

    X: np.ndarray
    valid_indices: List[int]
    invalid_indices: List[int]
    valid_smiles: List[str]


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    """
    Convert a SMILES string into an RDKit Mol object.

    Parameters
    ----------
    smiles : str
        Molecular SMILES representation.

    Returns
    -------
    mol : rdkit.Chem.Mol or None
        Parsed molecule object, or None if parsing fails.
    """
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def mol_to_morgan_fingerprint(
    mol: Chem.Mol,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Convert an RDKit Mol object into a Morgan fingerprint bit vector.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Valid RDKit molecule.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Number of fingerprint bits.

    Returns
    -------
    fp_array : np.ndarray
        1D binary fingerprint array of shape (n_bits,).
    """
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
    )

    fp_array = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bit_vector, fp_array)

    return fp_array


def featurize_smiles_list(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> FingerprintResult:
    """
    Featurize a sequence of SMILES strings into Morgan fingerprints.

    This function is reusable for:
    - training data
    - validation/test data
    - new inference-time molecules
    - future Streamlit/CLI inputs

    Parameters
    ----------
    smiles_list : sequence of str
        Input SMILES strings.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Number of bits in fingerprint vector.

    Returns
    -------
    FingerprintResult
        Featurization results including valid/invalid indices.
    """
    fingerprints: List[np.ndarray] = []
    valid_indices: List[int] = []
    invalid_indices: List[int] = []
    valid_smiles: List[str] = []

    for idx, smiles in enumerate(smiles_list):
        mol = smiles_to_mol(smiles)

        if mol is None:
            invalid_indices.append(idx)
            continue

        fp_array = mol_to_morgan_fingerprint(
            mol=mol,
            radius=radius,
            n_bits=n_bits,
        )

        fingerprints.append(fp_array)
        valid_indices.append(idx)
        valid_smiles.append(smiles)

    if fingerprints:
        X = np.vstack(fingerprints).astype(np.int8)
    else:
        X = np.empty((0, n_bits), dtype=np.int8)

    return FingerprintResult(
        X=X,
        valid_indices=valid_indices,
        invalid_indices=invalid_indices,
        valid_smiles=valid_smiles,
    )


def featurize_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> FingerprintResult:
    """
    Featurize SMILES stored in a dataframe column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing SMILES strings.
    smiles_column : str, default="smiles"
        Column containing SMILES.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Number of fingerprint bits.

    Returns
    -------
    FingerprintResult
        Featurization results aligned to dataframe row positions.

    Raises
    ------
    ValueError
        If the SMILES column is not present.
    """
    if smiles_column not in df.columns:
        raise ValueError(
            f"Column '{smiles_column}' not found in dataframe."
        )

    smiles_values = df[smiles_column].tolist()

    return featurize_smiles_list(
        smiles_list=smiles_values,
        radius=radius,
        n_bits=n_bits,
    )


def filter_dataframe_to_valid_rows(
    df: pd.DataFrame,
    valid_indices: Iterable[int],
) -> pd.DataFrame:
    """
    Keep only rows corresponding to successfully featurized molecules.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    valid_indices : iterable of int
        Row positions that produced valid fingerprints.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe, reset to a clean index.
    """
    valid_indices = list(valid_indices)
    return df.iloc[valid_indices].reset_index(drop=True)


def build_feature_matrix_and_targets(
    df: pd.DataFrame,
    label_columns: List[str],
    smiles_column: str = "smiles",
    radius: int = 2,
    n_bits: int = 2048,
    drop_rows_with_all_missing_labels: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, FingerprintResult]:
    """
    Build feature matrix X plus aligned targets/metadata from a dataframe.

    This is a high-value helper for the training pipeline because it preserves:
    - feature matrix
    - aligned labels
    - aligned metadata (mol_id, smiles)
    - invalid SMILES tracking

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_columns : list[str]
        Toxicity assay columns.
    smiles_column : str, default="smiles"
        Column containing SMILES.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Number of fingerprint bits.
    drop_rows_with_all_missing_labels : bool, default=False
        Whether to remove rows where all assay labels are missing
        after valid SMILES filtering.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y_df : pd.DataFrame
        Aligned target dataframe.
    metadata_df : pd.DataFrame
        Aligned metadata dataframe with mol_id and smiles when available.
    fp_result : FingerprintResult
        Full featurization result object.

    Raises
    ------
    ValueError
        If any requested label columns are missing.
    """
    missing_labels = [col for col in label_columns if col not in df.columns]
    if missing_labels:
        raise ValueError(
            f"Missing label columns: {missing_labels}"
        )

    fp_result = featurize_dataframe(
        df=df,
        smiles_column=smiles_column,
        radius=radius,
        n_bits=n_bits,
    )

    valid_df = filter_dataframe_to_valid_rows(df, fp_result.valid_indices)

    if drop_rows_with_all_missing_labels:
        keep_mask = ~valid_df[label_columns].isna().all(axis=1)
        valid_df = valid_df.loc[keep_mask].reset_index(drop=True)
        X = fp_result.X[keep_mask.to_numpy()]
    else:
        X = fp_result.X

    y_df = valid_df[label_columns].copy()

    metadata_columns = [col for col in ["mol_id", smiles_column] if col in valid_df.columns]
    metadata_df = valid_df[metadata_columns].copy()

    return X, y_df, metadata_df, fp_result


def featurize_single_smiles(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Featurize a single SMILES string for inference.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Number of fingerprint bits.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (1, n_bits).

    Raises
    ------
    ValueError
        If the SMILES string is invalid.
    """
    result = featurize_smiles_list(
        smiles_list=[smiles],
        radius=radius,
        n_bits=n_bits,
    )

    if len(result.valid_indices) == 0:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return result.X


def summarize_featurization(
    fp_result: FingerprintResult,
) -> dict:
    """
    Summarize featurization success/failure counts.

    Parameters
    ----------
    fp_result : FingerprintResult
        Result object from featurization.

    Returns
    -------
    dict
        Summary stats.
    """
    n_valid = len(fp_result.valid_indices)
    n_invalid = len(fp_result.invalid_indices)
    n_total = n_valid + n_invalid

    invalid_rate = (n_invalid / n_total) if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_rate": invalid_rate,
        "feature_shape": tuple(fp_result.X.shape),
    }