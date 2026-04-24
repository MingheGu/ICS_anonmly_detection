from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def bin_port(port: int) -> str:
    if port == 502:
        return "502"
    if port == -1:
        return "none"
    return "other"


FEATURE_SETS = {
    "fc_address": {
        "feature_columns": ["function_code", "address"],
        "categorical_columns": ["function_code", "address"],
        "numeric_columns": [],
    },
    "fc_address_ip": {
        "feature_columns": ["function_code", "address", "src_ip"],
        "categorical_columns": ["function_code", "address", "src_ip"],
        "numeric_columns": [],
    },
    "packet_full": {
        "feature_columns": [
            "function_code",
            "address",
            "src_ip",
            "dst_port_bin",
            "protocol",
            "tcp_flags",
            "payload_len",
        ],
        "categorical_columns": [
            "function_code",
            "address",
            "src_ip",
            "dst_port_bin",
            "protocol",
            "tcp_flags",
        ],
        "numeric_columns": ["payload_len"],
    },
    "fc_address_protocol_port": {
        "feature_columns": ["function_code", "address", "protocol", "dst_port_bin"],
        "categorical_columns": ["function_code", "address", "protocol", "dst_port_bin"],
        "numeric_columns": [],
    },
}


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns needed for per-packet models."""
    out = df.copy()
    out["dst_port_bin"] = out["dst_port"].apply(bin_port)
    return out


def build_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of train/validation/test packet splits is empty.")
    return train_df, val_df, test_df


def resolve_feature_set(feature_set: str) -> tuple[list[str], list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unsupported feature_set={feature_set}")
    config = FEATURE_SETS[feature_set]
    return (
        list(config["feature_columns"]),
        list(config["categorical_columns"]),
        list(config["numeric_columns"]),
    )


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False),
                categorical_columns,
            )
        )
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_metrics(
    labels: np.ndarray, pred: np.ndarray, scores: np.ndarray,
) -> dict[str, float | int]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    roc_auc = float("nan")
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = float(auc(fpr, tpr))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "roc_auc": roc_auc,
    }


def find_f1_optimal_threshold(
    labels: np.ndarray, scores: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (best_threshold, precision, recall, f1) from the PR curve."""
    prec_arr, rec_arr, thresholds = precision_recall_curve(labels, scores)
    f1_arr = np.where(
        (prec_arr[:-1] + rec_arr[:-1]) > 0,
        2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_arr))
    return (
        float(thresholds[best_idx]),
        float(prec_arr[best_idx]),
        float(rec_arr[best_idx]),
        float(f1_arr[best_idx]),
    )


def compute_threshold_from_scores(
    scores: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if len(scores) == 0:
        raise ValueError("Validation score set is empty.")
    quantile_threshold = float(np.quantile(scores, args.validation_quantile))
    if args.threshold_method == "sigma":
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        threshold = mean_s + args.threshold_sigma * std_s
        return threshold, quantile_threshold
    threshold = float(max(quantile_threshold, 1e-6))
    return threshold, quantile_threshold


def save_common_outputs(
    output_dir: Path,
    prefix: str,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_pred: np.ndarray,
    metrics_row: dict[str, float | int | str],
    test_scores_df: pd.DataFrame,
    val_scores_df: pd.DataFrame,
    run_summary: dict[str, object],
) -> None:
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)

    cm = confusion_matrix(test_labels, test_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["normal", "anomaly"], columns=["pred_normal", "pred_anomaly"]).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv"
    )

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(prefix.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()

    if len(np.unique(test_labels)) > 1:
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_auc = float(auc(fpr, tpr))
        plt.figure(figsize=(6, 4.5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{prefix.replace('_', ' ').title()} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4.5))
        PrecisionRecallDisplay.from_predictions(test_labels, test_scores)
        plt.title(f"{prefix.replace('_', ' ').title()} Precision-Recall Curve")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_pr_curve.png", dpi=200)
        plt.close()

    test_scores_df.to_csv(output_dir / f"{prefix}_test_scores.csv", index=False)
    val_scores_df.to_csv(output_dir / f"{prefix}_validation_scores.csv", index=False)
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-dataset",
        default="artifacts/rolling_fc_address/rolling_packet_features_fc_address.csv",
        help="Packet-level CSV generated by pre_process_rolling_fc_address.py",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address/packet_results_ocsvm_per_packet",
        help="Directory for per-packet One-Class SVM outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-set",
        default="fc_address_ip",
        choices=sorted(FEATURE_SETS.keys()),
        help="Which per-packet feature subset to use. "
             "fc_address_ip is recommended when scans and writes share Modbus/TCP control traffic.",
    )
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument(
        "--threshold-method",
        default="sigma",
        choices=["quantile", "sigma"],
        help="'sigma' sets threshold = mean + k*std of validation scores "
             "(recommended when normal scores pile up at a few discrete values); "
             "'quantile' uses the validation_quantile percentile.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Number of standard deviations above the validation mean to use as threshold "
             "(--threshold-method sigma only).",
    )
    parser.add_argument(
        "--validation-quantile",
        type=float,
        default=0.99,
        help="Quantile of validation anomaly score used as threshold.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / args.packet_dataset
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = prepare_df(df)
    train_df, val_df, test_df = build_splits(df)

    feature_columns, categorical_columns, numeric_columns = resolve_feature_set(args.feature_set)
    preprocessor = build_preprocessor(categorical_columns, numeric_columns)
    train_x = preprocessor.fit_transform(train_df[feature_columns])
    val_x = preprocessor.transform(val_df[feature_columns])
    test_x = preprocessor.transform(test_df[feature_columns])

    print(f"Feature dimensions after encoding: {train_x.shape[1]}")
    print(f"Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}")

    model = OneClassSVM(kernel=args.kernel, gamma=args.gamma, nu=args.nu)
    model.fit(train_x)

    # Use decision_function: positive = normal, negative = anomaly
    # Invert sign so higher = more anomalous (for threshold/AUC computation)
    val_scores = -model.decision_function(val_x)
    test_scores = -model.decision_function(test_x)
    val_labels = val_df["is_attack"].astype(int).to_numpy()
    test_labels = test_df["is_attack"].astype(int).to_numpy()

    # --- Method 1: Native boundary (decision_function < 0 → anomaly) ---
    native_pred = (test_scores > 0).astype(int)
    native_metrics = compute_metrics(test_labels, native_pred, test_scores)

    # --- Method 2: Validation threshold [PRIMARY] ---
    threshold, quantile_threshold = compute_threshold_from_scores(val_scores, args)
    primary_pred = (test_scores >= threshold).astype(int)
    primary_metrics = compute_metrics(test_labels, primary_pred, test_scores)

    # --- Diagnostic: validation quantile threshold ---
    quantile_threshold = float(np.quantile(val_scores, args.validation_quantile))
    quantile_pred = (test_scores >= quantile_threshold).astype(int)
    quantile_metrics = compute_metrics(test_labels, quantile_pred, test_scores)

    # --- Method 3: Val F1-optimal threshold [secondary diagnostic] ---
    val_threshold, val_prec, val_rec, val_f1 = find_f1_optimal_threshold(val_labels, val_scores)
    val_pred = (test_scores >= val_threshold).astype(int)
    val_metrics = compute_metrics(test_labels, val_pred, test_scores)

    # --- Method 4: Test F1-optimal threshold [ORACLE — uses test labels, upper bound only] ---
    oracle_threshold, _, _, _ = find_f1_optimal_threshold(test_labels, test_scores)
    oracle_pred = (test_scores >= oracle_threshold).astype(int)
    oracle_metrics = compute_metrics(test_labels, oracle_pred, test_scores)

    print("\n=== Native boundary (nu={}) [no labels] ===".format(args.nu))
    print(f"  Precision: {native_metrics['precision']:.4f}")
    print(f"  Recall:    {native_metrics['recall']:.4f}")
    print(f"  F1:        {native_metrics['f1_score']:.4f}")
    print(f"  TP={native_metrics['tp']} FP={native_metrics['fp']} TN={native_metrics['tn']} FN={native_metrics['fn']}")

    print(
        f"\n=== Validation {args.threshold_method} threshold "
        f"({threshold:.4f}) [PRIMARY] ==="
    )
    print(f"  Precision: {primary_metrics['precision']:.4f}")
    print(f"  Recall:    {primary_metrics['recall']:.4f}")
    print(f"  F1:        {primary_metrics['f1_score']:.4f}")
    print(
        f"  TP={primary_metrics['tp']} FP={primary_metrics['fp']} "
        f"TN={primary_metrics['tn']} FN={primary_metrics['fn']}"
    )

    print(
        f"\n=== Val quantile threshold p{args.validation_quantile * 100:.0f} "
        f"({quantile_threshold:.4f}) [secondary diagnostic] ==="
    )
    print(f"  Precision: {quantile_metrics['precision']:.4f}")
    print(f"  Recall:    {quantile_metrics['recall']:.4f}")
    print(f"  F1:        {quantile_metrics['f1_score']:.4f}")
    print(
        f"  TP={quantile_metrics['tp']} FP={quantile_metrics['fp']} "
        f"TN={quantile_metrics['tn']} FN={quantile_metrics['fn']}"
    )

    print(f"\n=== Val F1-optimal threshold ({val_threshold:.4f}) [secondary diagnostic] ===")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1:        {val_metrics['f1_score']:.4f}")
    print(f"  TP={val_metrics['tp']} FP={val_metrics['fp']} TN={val_metrics['tn']} FN={val_metrics['fn']}")

    print(f"\n=== Test oracle threshold ({oracle_threshold:.4f}) [ORACLE — uses test labels, upper bound only] ===")
    print(f"  Precision: {oracle_metrics['precision']:.4f}")
    print(f"  Recall:    {oracle_metrics['recall']:.4f}")
    print(f"  F1:        {oracle_metrics['f1_score']:.4f}")
    print(f"  TP={oracle_metrics['tp']} FP={oracle_metrics['fp']} TN={oracle_metrics['tn']} FN={oracle_metrics['fn']}")

    print(f"\n  AUC: {primary_metrics['roc_auc']:.4f}")

    metrics_row = {
        "model": "One-Class SVM",
        "threshold_method": args.threshold_method,
        "threshold": threshold,
        **primary_metrics,
        "validation_quantile": args.validation_quantile,
        "validation_quantile_threshold": quantile_threshold,
        "validation_quantile_precision": quantile_metrics["precision"],
        "validation_quantile_recall": quantile_metrics["recall"],
        "validation_quantile_f1": quantile_metrics["f1_score"],
        "threshold_sigma": args.threshold_sigma,
        "native_precision": native_metrics["precision"],
        "native_recall": native_metrics["recall"],
        "native_f1": native_metrics["f1_score"],
        "val_f1_threshold": val_threshold,
        "val_f1_precision": val_metrics["precision"],
        "val_f1_recall": val_metrics["recall"],
        "val_f1_f1": val_metrics["f1_score"],
        "oracle_f1_threshold": oracle_threshold,
        "oracle_precision": oracle_metrics["precision"],
        "oracle_recall": oracle_metrics["recall"],
        "oracle_f1": oracle_metrics["f1_score"],
        "kernel": args.kernel,
        "gamma": args.gamma,
        "nu": args.nu,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_set": args.feature_set,
        "feature_columns": ",".join(feature_columns),
        "feature_dims": int(train_x.shape[1]),
    }

    test_scores_df = test_df.reset_index(drop=True).copy()
    test_scores_df["anomaly_score"] = test_scores
    test_scores_df["pred_is_anomaly"] = primary_pred
    test_scores_df["pred_primary"] = primary_pred
    test_scores_df["pred_validation_quantile"] = quantile_pred
    test_scores_df["pred_native"] = native_pred
    test_scores_df["pred_val_f1"] = val_pred
    test_scores_df["pred_oracle_f1"] = oracle_pred
    val_scores_df = val_df.reset_index(drop=True).copy()
    val_scores_df["anomaly_score"] = val_scores

    run_summary = {
        "packet_dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "model": "One-Class SVM",
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "kernel": args.kernel,
        "gamma": args.gamma,
        "nu": args.nu,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_dims": int(train_x.shape[1]),
        "roc_auc": primary_metrics["roc_auc"],
        "threshold_method": args.threshold_method,
        "threshold_sigma": args.threshold_sigma,
        "validation_quantile": args.validation_quantile,
        "primary_threshold": threshold,
        "primary_metrics": primary_metrics,
        "validation_quantile_threshold (secondary diagnostic)": quantile_threshold,
        "validation_quantile_metrics (secondary diagnostic)": quantile_metrics,
        "val_f1_optimal_threshold": val_threshold,
        "val_f1_optimal_metrics (secondary diagnostic)": val_metrics,
        "native_metrics": native_metrics,
        "oracle_f1_optimal_threshold": oracle_threshold,
        "oracle_f1_optimal_metrics (uses test labels, upper bound only)": oracle_metrics,
    }

    save_common_outputs(
        output_dir=output_dir,
        prefix="packet_ocsvm",
        test_labels=test_labels,
        test_scores=test_scores,
        test_pred=primary_pred,
        metrics_row=metrics_row,
        test_scores_df=test_scores_df,
        val_scores_df=val_scores_df,
        run_summary=run_summary,
    )

    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
