from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.load_data import load_tox21_data, summarize_dataset
from src.featurization.fingerprints import (
    build_feature_matrix_and_targets,
    summarize_featurization,
)
from src.models.train_logreg import (
    extract_trained_models,
    predict_with_trained_models,
    summarize_training_results,
    train_logreg_multitask,
)
from src.evaluation.metrics import (
    build_overall_metrics_summary,
    evaluate_multitask_predictions,
)
from src.evaluation.plots import generate_standard_evaluation_plots
from src.evaluation.error_analysis import (
    build_prediction_table,
    save_error_analysis_tables,
)


def ensure_project_dirs() -> dict[str, Path]:
    """
    Create standard project output directories.

    Returns
    -------
    dict[str, Path]
        Dictionary of key directory paths.
    """
    paths = {
        "outputs": Path("outputs"),
        "metrics": Path("outputs/metrics"),
        "plots": Path("outputs/plots"),
        "predictions": Path("outputs/predictions"),
        "error_analysis": Path("outputs/error_analysis"),
        "models_saved": Path("models_saved"),
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save dataframe to CSV.
    """
    df.to_csv(path, index=False)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    dirs = ensure_project_dirs()

    DATA_PATH = "data/tox21.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Morgan fingerprint settings
    RADIUS = 2
    N_BITS = 2048

    # Logistic Regression settings
    LOGREG_MAX_ITER = 1000
    LOGREG_CLASS_WEIGHT = "balanced"
    LOGREG_SOLVER = "liblinear"
    LOGREG_C = 1.0

    print("=" * 70)
    print("TOX21 TOXICITY PREDICTION PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading dataset...")
    df, label_columns = load_tox21_data(
        file_path=DATA_PATH,
        drop_missing_smiles=True,
        drop_duplicate_smiles=True,
        label_handling="keep",
    )

    dataset_summary = summarize_dataset(df, label_columns)
    dataset_summary_df = pd.DataFrame([dataset_summary])
    save_dataframe(dataset_summary_df, dirs["metrics"] / "dataset_summary.csv")

    print(f"Loaded dataset with {dataset_summary['n_rows']} rows")
    print(f"Detected {dataset_summary['n_labels']} assay columns")

    # ------------------------------------------------------------------
    # 3. Featurize data
    # ------------------------------------------------------------------
    print("\n[2/8] Generating Morgan fingerprints...")
    X, y_df, metadata_df, fp_result = build_feature_matrix_and_targets(
        df=df,
        label_columns=label_columns,
        smiles_column="smiles",
        radius=RADIUS,
        n_bits=N_BITS,
        drop_rows_with_all_missing_labels=True,
    )

    featurization_summary = summarize_featurization(fp_result)
    featurization_summary_df = pd.DataFrame([featurization_summary])
    save_dataframe(featurization_summary_df, dirs["metrics"] / "featurization_summary.csv")

    print(f"Feature matrix shape: {featurization_summary['feature_shape']}")
    print(f"Valid molecules: {featurization_summary['n_valid']}")
    print(f"Invalid molecules: {featurization_summary['n_invalid']}")

    # ------------------------------------------------------------------
    # 4. Train/test split
    # ------------------------------------------------------------------
    print("\n[3/8] Splitting into train/test sets...")
    (
        X_train,
        X_test,
        y_train_df,
        y_test_df,
        metadata_train_df,
        metadata_test_df,
    ) = train_test_split(
        X,
        y_df,
        metadata_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    y_train_df = y_train_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)
    metadata_train_df = metadata_train_df.reset_index(drop=True)
    metadata_test_df = metadata_test_df.reset_index(drop=True)

    print(f"Training samples: {len(y_train_df)}")
    print(f"Test samples: {len(y_test_df)}")

    split_summary_df = pd.DataFrame(
        [
            {
                "n_total": len(y_df),
                "n_train": len(y_train_df),
                "n_test": len(y_test_df),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        ]
    )
    save_dataframe(split_summary_df, dirs["metrics"] / "split_summary.csv")

    # ------------------------------------------------------------------
    # 5. Train baseline model
    # ------------------------------------------------------------------
    print("\n[4/8] Training Logistic Regression baseline...")
    training_results = train_logreg_multitask(
        X=X_train,
        y_df=y_train_df,
        assay_names=label_columns,
        random_state=RANDOM_STATE,
        max_iter=LOGREG_MAX_ITER,
        class_weight=LOGREG_CLASS_WEIGHT,
        solver=LOGREG_SOLVER,
        C=LOGREG_C,
    )

    training_summary_df = summarize_training_results(training_results)
    save_dataframe(training_summary_df, dirs["metrics"] / "training_summary_logreg.csv")

    trained_models = extract_trained_models(training_results)

    print(f"Successfully trained {len(trained_models)} assay models out of {len(label_columns)}")

    if len(trained_models) == 0:
        raise RuntimeError("No assay models were successfully trained.")

    trained_assays = list(trained_models.keys())

    # ------------------------------------------------------------------
    # 6. Predict on test set
    # ------------------------------------------------------------------
    print("\n[5/8] Generating test predictions...")
    y_pred_df, y_proba_df = predict_with_trained_models(
        trained_models=trained_models,
        X=X_test,
    )

    # restrict truth to trained assays only
    y_test_eval_df = y_test_df[trained_assays].copy()

    save_dataframe(y_pred_df, dirs["predictions"] / "test_predictions_binary.csv")
    save_dataframe(y_proba_df, dirs["predictions"] / "test_predictions_proba.csv")
    save_dataframe(metadata_test_df, dirs["predictions"] / "test_metadata.csv")
    save_dataframe(y_test_eval_df, dirs["predictions"] / "test_true_labels.csv")

    # ------------------------------------------------------------------
    # 7. Evaluate metrics
    # ------------------------------------------------------------------
    print("\n[6/8] Evaluating model performance...")
    per_assay_metrics_df = evaluate_multitask_predictions(
        y_true_df=y_test_eval_df,
        y_pred_df=y_pred_df,
        y_proba_df=y_proba_df,
        assay_names=trained_assays,
    )

    overall_metrics_df = build_overall_metrics_summary(
        per_assay_df=per_assay_metrics_df,
        y_true_df=y_test_eval_df,
        y_pred_df=y_pred_df,
        y_proba_df=y_proba_df,
        assay_names=trained_assays,
    )

    save_dataframe(per_assay_metrics_df, dirs["metrics"] / "per_assay_metrics_logreg.csv")
    save_dataframe(overall_metrics_df, dirs["metrics"] / "overall_metrics_logreg.csv")

    print("Saved per-assay and overall metrics.")

    # ------------------------------------------------------------------
    # 8. Generate plots
    # ------------------------------------------------------------------
    print("\n[7/8] Generating evaluation plots...")
    plot_paths = generate_standard_evaluation_plots(
        y_true_df=y_test_eval_df,
        y_proba_df=y_proba_df,
        per_assay_df=per_assay_metrics_df,
        assay_names=trained_assays,
        output_dir=dirs["plots"],
    )

    plot_summary_rows = []
    for key, value in plot_paths.items():
        if isinstance(value, list):
            for path in value:
                plot_summary_rows.append({"plot_group": key, "path": str(path)})
        else:
            plot_summary_rows.append({"plot_group": key, "path": str(value)})

    plot_summary_df = pd.DataFrame(plot_summary_rows)
    save_dataframe(plot_summary_df, dirs["plots"] / "plot_manifest.csv")

    print("Saved evaluation plots.")

    # ------------------------------------------------------------------
    # 9. Error analysis
    # ------------------------------------------------------------------
    print("\n[8/8] Running error analysis...")
    prediction_table = build_prediction_table(
        metadata_df=metadata_test_df,
        y_true_df=y_test_eval_df,
        y_pred_df=y_pred_df,
        y_proba_df=y_proba_df,
        assay_names=trained_assays,
    )

    save_dataframe(prediction_table, dirs["predictions"] / "prediction_table_long_format.csv")

    error_files = save_error_analysis_tables(
        prediction_table=prediction_table,
        output_dir=dirs["error_analysis"],
        top_n=20,
    )

    error_manifest_df = pd.DataFrame(
        [{"table_name": name, "path": str(path)} for name, path in error_files.items()]
    )
    save_dataframe(error_manifest_df, dirs["error_analysis"] / "error_manifest.csv")

    print("Saved error-analysis tables.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print("\nTop-level outputs:")
    print(f"- Metrics: {dirs['metrics']}")
    print(f"- Plots: {dirs['plots']}")
    print(f"- Predictions: {dirs['predictions']}")
    print(f"- Error analysis: {dirs['error_analysis']}")

    print("\nOverall metrics:")
    print(overall_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()