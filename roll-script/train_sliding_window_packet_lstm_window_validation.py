from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
OLD_TRAIN_DIR = SCRIPT_DIR / "old_roll_train"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if OLD_TRAIN_DIR.exists() and str(OLD_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_TRAIN_DIR))

from train_roll_packet_lstm_v2 import (  # type: ignore
    PacketRollLSTM,
    anomaly_scores,
    build_packet_samples,
    build_token_mapping,
    cross_entropy_loss,
    set_seed,
    smooth_scores,
)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_time_windows(
    df: pd.DataFrame,
    train_duration_s: float,
    test_duration_s: float,
    step_s: float,
) -> list[dict[str, float]]:
    start_time = float(df["time_offset_s"].min())
    end_time = float(df["time_offset_s"].max())
    total_duration = train_duration_s + test_duration_s
    windows: list[dict[str, float]] = []
    current_start = start_time

    while current_start + total_duration <= end_time:
        train_start = round(current_start, 6)
        train_end = round(train_start + train_duration_s, 6)
        test_start = train_end
        test_end = round(test_start + test_duration_s, 6)
        windows.append(
            {
                "train_start_s": train_start,
                "train_end_s": train_end,
                "test_start_s": test_start,
                "test_end_s": test_end,
            }
        )
        current_start += step_s

    return windows


def build_fixed_train_windows(
    df: pd.DataFrame,
    fixed_train_end_s: float,
    test_duration_s: float,
    step_s: float,
) -> list[dict[str, float]]:
    """Train window is fixed at [0, fixed_train_end_s]. Only test window slides."""
    end_time = float(df["time_offset_s"].max())
    test_start = fixed_train_end_s
    windows: list[dict[str, float]] = []
    while test_start + test_duration_s <= end_time:
        windows.append({
            "train_start_s": 0.0,
            "train_end_s": fixed_train_end_s,
            "test_start_s": round(test_start, 6),
            "test_end_s": round(test_start + test_duration_s, 6),
        })
        test_start += step_s
    return windows


def slice_by_time(df: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    return df[(df["time_offset_s"] >= start_s) & (df["time_offset_s"] < end_s)].copy()


def filter_packet_rows(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows = int(len(df))
    filtered_df = df.copy()

    if args.exclude_tcp_fc_minus1:
        filtered_df = filtered_df[
            ~((filtered_df["protocol"] == "TCP") & (filtered_df["function_code"].astype(int) == -1))
        ].copy()

    summary = {
        "original_rows": original_rows,
        "filtered_rows": int(len(filtered_df)),
    }
    return filtered_df, summary


def create_model(args: argparse.Namespace) -> nn.Module:
    return PacketRollLSTM(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(resolve_device())


def fit_model(
    model: nn.Module,
    train_contexts: np.ndarray,
    train_targets: np.ndarray,
    val_contexts: np.ndarray,
    val_targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    if len(train_contexts) == 0 or len(val_contexts) == 0:
        raise ValueError("Train or validation sample set is empty.")

    device = resolve_device()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_contexts), torch.tensor(train_targets)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.long)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = cross_entropy_loss(model, val_contexts, val_targets, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "device": str(device),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_run": int(len(train_losses)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }


def build_test_samples_with_context(
    df: pd.DataFrame,
    test_start_s: float,
    test_end_s: float,
    context_length: int,
    token_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    time_values = df["time_offset_s"].to_numpy(dtype=float)
    first_test_idx = int(np.searchsorted(time_values, test_start_s, side="left"))
    last_test_idx = int(np.searchsorted(time_values, test_end_s, side="left"))
    if first_test_idx >= last_test_idx:
        return np.array([]), np.array([]), pd.DataFrame()

    buffer_start_idx = max(0, first_test_idx - context_length)
    buffered_df = df.iloc[buffer_start_idx:last_test_idx].copy()
    contexts, targets, metadata = build_packet_samples(buffered_df, context_length, token_to_idx)
    if len(contexts) == 0:
        return contexts, targets, pd.DataFrame()

    metadata_df = pd.DataFrame(metadata)
    actual_mask = (
        (metadata_df["time_offset_s"] >= test_start_s)
        & (metadata_df["time_offset_s"] < test_end_s)
    ).to_numpy()
    return contexts[actual_mask], targets[actual_mask], metadata_df.loc[actual_mask].reset_index(drop=True)


def score_samples(
    model: nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    raw_scores = anomaly_scores(model, contexts, targets, device, batch_size=args.score_batch_size)
    return smooth_scores(raw_scores, args.smooth_window)


def compute_threshold_from_scores(scores: np.ndarray, args: argparse.Namespace) -> tuple[float, float]:
    if len(scores) == 0:
        raise ValueError("Validation score set is empty.")
    if args.threshold_method == "sigma":
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        threshold = mean_s + args.threshold_sigma * std_s
        return threshold, threshold
    quantile_threshold = float(np.quantile(scores, args.validation_quantile))
    threshold = float(max(quantile_threshold, 1e-6))
    return threshold, quantile_threshold


def compute_metrics_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    roc_auc = float("nan")
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = float(auc(fpr, tpr))

    return pred, {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "roc_auc": roc_auc,
    }


def remove_high_score_targets(
    train_df: pd.DataFrame,
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    contexts, targets, metadata = build_packet_samples(train_df, args.context_length, token_to_idx)
    if len(contexts) == 0:
        return train_df

    split_idx = max(1, int(len(contexts) * (1.0 - args.val_fraction)))
    split_idx = min(split_idx, len(contexts) - 1) if len(contexts) > 1 else 1
    if split_idx <= 0 or split_idx >= len(contexts):
        return train_df

    model = create_model(args)
    model, summary = fit_model(
        model,
        contexts[:split_idx],
        targets[:split_idx],
        contexts[split_idx:],
        targets[split_idx:],
        args,
    )
    device = torch.device(str(summary["device"]))
    raw_scores = anomaly_scores(model, contexts, targets, device, batch_size=args.score_batch_size)
    scores = smooth_scores(raw_scores, args.smooth_window)
    cutoff = float(np.quantile(scores, args.self_clean_quantile))
    metadata_df = pd.DataFrame(metadata)
    keep_offsets = set(metadata_df.loc[scores < cutoff, "time_offset_s"].round(6).tolist())

    cleaned_df = train_df[
        (~train_df["time_offset_s"].round(6).isin(metadata_df["time_offset_s"].round(6)))
        | (train_df["time_offset_s"].round(6).isin(keep_offsets))
    ].copy()
    return cleaned_df


def contiguous_attack_segments(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    attack_df = df[df["is_attack"] == 1].sort_values("time_offset_s").reset_index(drop=True)
    if attack_df.empty:
        return []

    segments: list[tuple[float, float, str]] = []
    current_label = str(attack_df.loc[0, "label"])
    segment_start = float(attack_df.loc[0, "time_offset_s"])
    previous_time = segment_start

    for _, row in attack_df.iloc[1:].iterrows():
        current_time = float(row["time_offset_s"])
        label = str(row["label"])
        if label != current_label or current_time - previous_time > 10.0:
            segments.append((segment_start, previous_time, current_label))
            current_label = label
            segment_start = current_time
        previous_time = current_time

    segments.append((segment_start, previous_time, current_label))
    return segments


def run_one_window(
    df: pd.DataFrame,
    window_step: int,
    window: dict[str, float],
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, list[float], list[float]]:
    train_df = slice_by_time(df, window["train_start_s"], window["train_end_s"])
    if train_df.empty:
        return {
            "window_step": window_step,
            **window,
            "status": "skipped",
            "skip_reason": "empty_train_window",
        }, pd.DataFrame(), [], []

    val_cutoff_s = window["train_start_s"] + (window["train_end_s"] - window["train_start_s"]) * (1.0 - args.val_fraction)
    train_proper_df = train_df[train_df["time_offset_s"] < val_cutoff_s].copy()
    val_df = train_df[train_df["time_offset_s"] >= val_cutoff_s].copy()

    if args.oracle_clean_normal_only:
        train_proper_df = train_proper_df[train_proper_df["is_attack"] == 0].copy()
        val_df = val_df[val_df["is_attack"] == 0].copy()

    for _ in range(args.self_clean_rounds):
        train_proper_df = remove_high_score_targets(train_proper_df, token_to_idx, args)

    train_contexts, train_targets, _ = build_packet_samples(train_proper_df, args.context_length, token_to_idx)
    val_contexts, val_targets, _ = build_packet_samples(val_df, args.context_length, token_to_idx)
    test_contexts, test_targets, test_meta_df = build_test_samples_with_context(
        df,
        window["test_start_s"],
        window["test_end_s"],
        args.context_length,
        token_to_idx,
    )

    if len(train_contexts) == 0 or len(val_contexts) == 0 or len(test_contexts) == 0:
        return {
            "window_step": window_step,
            **window,
            "status": "skipped",
            "skip_reason": "insufficient_samples",
            "train_packets": int(len(train_proper_df)),
            "val_packets": int(len(val_df)),
            "test_packets": int(len(test_meta_df)),
        }, pd.DataFrame(), [], []

    model = create_model(args)
    model, train_summary = fit_model(model, train_contexts, train_targets, val_contexts, val_targets, args)
    device = torch.device(str(train_summary["device"]))
    val_scores = score_samples(model, val_contexts, val_targets, args, device)

    test_raw_scores = anomaly_scores(model, test_contexts, test_targets, device, batch_size=args.score_batch_size)
    test_scores = smooth_scores(test_raw_scores, args.smooth_window)
    if args.threshold_method == "test_quantile":
        threshold = float(np.quantile(test_scores, args.test_anomaly_quantile))
        quantile_threshold = threshold
    else:
        threshold, quantile_threshold = compute_threshold_from_scores(val_scores, args)
    test_labels = test_meta_df["is_attack"].astype(int).to_numpy()
    test_pred, metric_values = compute_metrics_at_threshold(test_labels, test_scores, threshold)

    scores_df = test_meta_df.copy()
    scores_df["window_step"] = window_step
    scores_df["train_start_s"] = window["train_start_s"]
    scores_df["train_end_s"] = window["train_end_s"]
    scores_df["test_start_s"] = window["test_start_s"]
    scores_df["test_end_s"] = window["test_end_s"]
    scores_df["raw_anomaly_score"] = test_raw_scores
    scores_df["anomaly_score"] = test_scores
    scores_df["pred_is_anomaly"] = test_pred
    scores_df["threshold"] = threshold

    metrics_row = {
        "window_step": window_step,
        **window,
        "status": "ok",
        "skip_reason": "",
        "train_packets": int(len(train_proper_df)),
        "val_packets": int(len(val_df)),
        "test_packets": int(len(test_meta_df)),
        "train_samples": int(len(train_contexts)),
        "val_samples": int(len(val_contexts)),
        "test_samples": int(len(test_contexts)),
        "train_attack_frac": float(train_proper_df["is_attack"].mean()) if len(train_proper_df) else 0.0,
        "test_attack_frac": float(test_meta_df["is_attack"].mean()) if len(test_meta_df) else 0.0,
        "threshold_calibration_mode": "test_quantile" if args.threshold_method == "test_quantile" else "per_window_validation",
        "threshold_calibration_split": "test_window" if args.threshold_method == "test_quantile" else "window_validation",
        "threshold": threshold,
        "quantile_threshold": quantile_threshold,
        "calib_score_mean": float(np.mean(val_scores)),
        "calib_score_std": float(np.std(val_scores)),
        "calib_score_p50": float(np.median(val_scores)),
        "calib_score_p99": float(np.quantile(val_scores, 0.99)),
        "epochs_run": int(train_summary["epochs_run"]),
        "best_epoch": int(train_summary["best_epoch"]),
        "best_val_loss": float(train_summary["best_val_loss"]),
        **metric_values,
    }
    return metrics_row, scores_df, list(train_summary["train_losses"]), list(train_summary["val_losses"])


def run_fixed_train_mode(
    df: pd.DataFrame,
    windows: list[dict[str, float]],
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[float], list[float]]:
    """Train once on fixed window, score each sliding test window without retraining."""
    if not windows:
        return [], pd.DataFrame(), [], []

    fixed_window = windows[0]
    train_df = slice_by_time(df, fixed_window["train_start_s"], fixed_window["train_end_s"])

    val_cutoff_s = fixed_window["train_start_s"] + (
        fixed_window["train_end_s"] - fixed_window["train_start_s"]
    ) * (1.0 - args.val_fraction)
    train_proper_df = train_df[train_df["time_offset_s"] < val_cutoff_s].copy()
    val_df = train_df[train_df["time_offset_s"] >= val_cutoff_s].copy()

    if args.oracle_clean_normal_only:
        train_proper_df = train_proper_df[train_proper_df["is_attack"] == 0].copy()
        val_df = val_df[val_df["is_attack"] == 0].copy()

    train_contexts, train_targets, _ = build_packet_samples(train_proper_df, args.context_length, token_to_idx)
    val_contexts, val_targets, _ = build_packet_samples(val_df, args.context_length, token_to_idx)

    model = create_model(args)
    model, train_summary = fit_model(model, train_contexts, train_targets, val_contexts, val_targets, args)
    device = torch.device(str(train_summary["device"]))

    val_scores = score_samples(model, val_contexts, val_targets, args, device)
    if args.threshold_method != "test_quantile":
        global_threshold, global_quantile_threshold = compute_threshold_from_scores(val_scores, args)

    metrics_rows: list[dict[str, Any]] = []
    all_scores: list[pd.DataFrame] = []

    for window_step, window in enumerate(windows, start=1):
        test_contexts, test_targets, test_meta_df = build_test_samples_with_context(
            df, window["test_start_s"], window["test_end_s"], args.context_length, token_to_idx,
        )
        if len(test_contexts) == 0:
            metrics_rows.append({
                "window_step": window_step, **window,
                "status": "skipped", "skip_reason": "empty_test",
            })
            continue

        test_raw_scores = anomaly_scores(model, test_contexts, test_targets, device, batch_size=args.score_batch_size)
        test_scores = smooth_scores(test_raw_scores, args.smooth_window)
        if args.threshold_method == "test_quantile":
            threshold = float(np.quantile(test_scores, args.test_anomaly_quantile))
            quantile_threshold = threshold
        else:
            threshold = global_threshold
            quantile_threshold = global_quantile_threshold
        test_labels = test_meta_df["is_attack"].astype(int).to_numpy()
        test_pred, metric_values = compute_metrics_at_threshold(test_labels, test_scores, threshold)

        scores_df = test_meta_df.copy()
        scores_df["window_step"] = window_step
        scores_df["train_start_s"] = fixed_window["train_start_s"]
        scores_df["train_end_s"] = fixed_window["train_end_s"]
        scores_df["test_start_s"] = window["test_start_s"]
        scores_df["test_end_s"] = window["test_end_s"]
        scores_df["raw_anomaly_score"] = test_raw_scores
        scores_df["anomaly_score"] = test_scores
        scores_df["pred_is_anomaly"] = test_pred
        scores_df["threshold"] = threshold
        all_scores.append(scores_df)

        metrics_rows.append({
            "window_step": window_step, **window,
            "status": "ok", "skip_reason": "",
            "train_packets": int(len(train_proper_df)),
            "val_packets": int(len(val_df)),
            "test_packets": int(len(test_meta_df)),
            "threshold": threshold,
            "quantile_threshold": quantile_threshold,
            "threshold_calibration_mode": "test_quantile" if args.threshold_method == "test_quantile" else "fixed_train",
            **metric_values,
        })

    combined_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    return metrics_rows, combined_scores, list(train_summary["train_losses"]), list(train_summary["val_losses"])


def plot_score_timeline(
    scores_df: pd.DataFrame,
    attack_segments: list[tuple[float, float, str]],
    output_path: Path,
) -> None:
    if scores_df.empty:
        return

    aggregated = (
        scores_df.groupby("time_offset_s", as_index=False)
        .agg(anomaly_score=("anomaly_score", "mean"), threshold=("threshold", "mean"))
        .sort_values("time_offset_s")
    )

    plt.figure(figsize=(12, 5))
    plt.plot(aggregated["time_offset_s"], aggregated["anomaly_score"], label="mean anomaly score", linewidth=1.2)
    plt.plot(aggregated["time_offset_s"], aggregated["threshold"], label="mean threshold", linewidth=1.0, linestyle="--")
    for start_s, end_s, label in attack_segments:
        plt.axvspan(start_s, end_s, alpha=0.18, color="tomato")
        plt.text(start_s, plt.ylim()[1] * 0.95, label, fontsize=8, va="top")
    plt.xlabel("time_offset_s")
    plt.ylabel("anomaly score")
    plt.title("Sliding Window Packet LSTM Score Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty:
        return
    plt.figure(figsize=(10, 4.5))
    plt.plot(metrics_df["window_step"], metrics_df["precision"], label="precision")
    plt.plot(metrics_df["window_step"], metrics_df["recall"], label="recall")
    plt.plot(metrics_df["window_step"], metrics_df["f1_score"], label="f1")
    plt.xlabel("window_step")
    plt.ylabel("metric")
    plt.title("Sliding Window Metrics Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_thresholds(metrics_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty:
        return
    plt.figure(figsize=(10, 4.5))
    plt.plot(metrics_df["window_step"], metrics_df["threshold"], label="threshold")
    plt.xlabel("window_step")
    plt.ylabel("threshold")
    plt.title("Sliding Window Threshold Over Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-dataset",
        default="artifacts/rolling_fc_address_port_protocol/rolling_packet_features_fc_address.csv",
        help="Packet-level rolling CSV generated by pre_process_rolling_fc_address.py",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address_port_protocol/sliding_window_window_validation_results",
        help="Directory for sliding-window packet-level LSTM outputs.",
    )
    parser.add_argument("--pcap-name", default="mixed_long_conti")
    parser.add_argument("--train-duration-s", type=float, default=900.0)
    parser.add_argument("--test-duration-s", type=float, default=300.0)
    parser.add_argument("--step-s", type=float, default=120.0)
    parser.add_argument(
        "--fixed-train-end-s",
        type=float,
        default=0.0,
        help="If > 0, enable fixed-train mode: train ONCE on [0, fixed_train_end_s], "
             "then slide only the test window. No retraining per window.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--oracle-clean-normal-only",
        action="store_true",
        help="Diagnostic mode: filter attack-labeled packets out of the training and window-validation slices.",
    )
    parser.add_argument("--self-clean-rounds", type=int, default=0)
    parser.add_argument("--self-clean-quantile", type=float, default=0.90)
    parser.add_argument("--context-length", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-method",
        default="sigma",
        choices=["quantile", "sigma", "test_quantile"],
        help="'sigma' sets threshold = mean + k*std of calibration scores (robust when normal scores cluster near zero); "
             "'quantile' uses the validation_quantile percentile (original behaviour); "
             "'test_quantile' sets a per-test-window threshold at the test score quantile.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Number of standard deviations above the calibration mean to use as threshold (--threshold-method sigma only).",
    )
    parser.add_argument(
        "--test-anomaly-quantile",
        type=float,
        default=0.99,
        help="Quantile of test scores used as threshold when --threshold-method test_quantile.",
    )
    parser.add_argument(
        "--exclude-tcp-fc-minus1",
        action="store_true",
        help="Drop packets where protocol == TCP and function_code == -1 before building windows.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / args.packet_dataset
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(dataset_path)
    df = full_df[full_df["pcap_name"] == args.pcap_name].copy()
    if df.empty:
        raise ValueError(f"No rows found for pcap_name={args.pcap_name}")

    df, filter_summary = filter_packet_rows(df, args)
    if df.empty:
        raise ValueError("No rows remain after packet filtering.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    token_to_idx = build_token_mapping(df)
    args.vocab_size = len(token_to_idx)
    if args.fixed_train_end_s > 0:
        windows = build_fixed_train_windows(
            df, args.fixed_train_end_s, args.test_duration_s, args.step_s
        )
    else:
        windows = build_time_windows(
            df, args.train_duration_s, args.test_duration_s, args.step_s
        )

    if args.fixed_train_end_s > 0:
        metrics_rows, scores_df_final, last_train_losses, last_val_losses = run_fixed_train_mode(
            df, windows, token_to_idx, args
        )
        metrics_df = pd.DataFrame(metrics_rows)
        scores_df = scores_df_final
    else:
        metrics_rows = []
        all_scores = []
        last_train_losses = []
        last_val_losses = []
        for window_step, window in enumerate(windows, start=1):
            metrics_row, scores_df_w, train_losses, val_losses = run_one_window(
                df, window_step, window, token_to_idx, args,
            )
            metrics_rows.append(metrics_row)
            if not scores_df_w.empty:
                all_scores.append(scores_df_w)
            if train_losses and val_losses:
                last_train_losses = train_losses
                last_val_losses = val_losses
        metrics_df = pd.DataFrame(metrics_rows)
        scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()

    metrics_path = output_dir / "sliding_window_step_metrics.csv"
    scores_path = output_dir / "sliding_window_all_scores.csv"
    metrics_df.to_csv(metrics_path, index=False)
    scores_df.to_csv(scores_path, index=False)

    attack_segments = contiguous_attack_segments(df)
    plot_score_timeline(scores_df, attack_segments, output_dir / "sliding_window_score_timeline.png")
    plot_metrics(metrics_df[metrics_df["status"] == "ok"], output_dir / "sliding_window_metrics_over_time.png")
    plot_thresholds(metrics_df[metrics_df["status"] == "ok"], output_dir / "sliding_window_threshold_over_time.png")

    if last_train_losses and last_val_losses:
        plt.figure(figsize=(7, 4))
        epochs = np.arange(1, len(last_train_losses) + 1)
        plt.plot(epochs, last_train_losses, label="train_loss")
        plt.plot(epochs, last_val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Last Window Training Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sliding_window_last_training_curve.png", dpi=200)
        plt.close()

    run_summary = {
        "packet_dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "pcap_name": args.pcap_name,
        "window_count": int(len(windows)),
        "successful_window_count": int((metrics_df["status"] == "ok").sum()),
        "train_duration_s": args.train_duration_s,
        "test_duration_s": args.test_duration_s,
        "step_s": args.step_s,
        "fixed_train_end_s": args.fixed_train_end_s,
        "val_fraction": args.val_fraction,
        "oracle_clean_normal_only": bool(args.oracle_clean_normal_only),
        "self_clean_rounds": args.self_clean_rounds,
        "self_clean_quantile": args.self_clean_quantile,
        "context_length": args.context_length,
        "smooth_window": args.smooth_window,
        "validation_quantile": args.validation_quantile,
        "test_anomaly_quantile": args.test_anomaly_quantile,
        "threshold_calibration_mode": (
            "test_quantile"
            if args.threshold_method == "test_quantile"
            else ("fixed_train" if args.fixed_train_end_s > 0 else "per_window_validation")
        ),
        "threshold_calibration_split": "test_window" if args.threshold_method == "test_quantile" else "window_validation",
        "vocab_size": len(token_to_idx),
        "exclude_tcp_fc_minus1": bool(args.exclude_tcp_fc_minus1),
        **filter_summary,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote scores to {scores_path}")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
