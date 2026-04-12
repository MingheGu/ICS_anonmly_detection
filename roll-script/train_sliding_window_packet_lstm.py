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


def slice_by_time(df: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    return df[(df["time_offset_s"] >= start_s) & (df["time_offset_s"] < end_s)].copy()


def create_model(args: argparse.Namespace) -> nn.Module:
    return PacketRollLSTM(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(torch.device("cpu"))


def fit_model(
    model: nn.Module,
    train_contexts: np.ndarray,
    train_targets: np.ndarray,
    val_contexts: np.ndarray,
    val_targets: np.ndarray,
    epochs: int,
    patience: int,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    if len(train_contexts) == 0 or len(val_contexts) == 0:
        raise ValueError("Train or validation sample set is empty for this fit.")

    device = torch.device("cpu")
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

    for epoch in range(epochs):
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
            if epochs_without_improvement >= patience:
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


def train_model(
    train_contexts: np.ndarray,
    train_targets: np.ndarray,
    val_contexts: np.ndarray,
    val_targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    model = create_model(args)
    return fit_model(
        model,
        train_contexts,
        train_targets,
        val_contexts,
        val_targets,
        args.epochs,
        args.patience,
        args,
    )


def score_samples(
    model: nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if len(contexts) == 0:
        raise ValueError("Sample set is empty.")
    raw_scores = anomaly_scores(model, contexts, targets, device, batch_size=args.score_batch_size)
    return smooth_scores(raw_scores, args.smooth_window)


def compute_threshold_from_scores(
    scores: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if len(scores) == 0:
        raise ValueError("Threshold calibration score set is empty.")
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

    model, summary = train_model(
        contexts[:split_idx],
        targets[:split_idx],
        contexts[split_idx:],
        targets[split_idx:],
        args,
    )
    device = torch.device(str(summary["device"]))
    raw_scores = anomaly_scores(model, contexts, targets, device, batch_size=args.score_batch_size)
    scores = smooth_scores(raw_scores, args.smooth_window)
    cutoff = float(np.quantile(scores, 0.95))
    metadata_df = pd.DataFrame(metadata)
    keep_offsets = set(
        metadata_df.loc[scores < cutoff, "time_offset_s"].round(6).tolist()
    )

    cleaned_df = train_df[
        (~train_df["time_offset_s"].round(6).isin(metadata_df["time_offset_s"].round(6)))
        | (train_df["time_offset_s"].round(6).isin(keep_offsets))
    ].copy()
    return cleaned_df


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
    calibration_contexts: np.ndarray | None,
    calibration_targets: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, list[float], list[float]]:
    train_df = slice_by_time(df, window["train_start_s"], window["train_end_s"])
    if train_df.empty:
        return {
            "window_step": window_step,
            **window,
            "status": "skipped",
            "skip_reason": "empty_train_window",
        }, pd.DataFrame(), np.array([]), [], []

    val_cutoff_s = window["train_start_s"] + (window["train_end_s"] - window["train_start_s"]) * (1.0 - args.val_fraction)
    train_proper_df = train_df[train_df["time_offset_s"] < val_cutoff_s].copy()
    val_df = train_df[train_df["time_offset_s"] >= val_cutoff_s].copy()

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
        }, pd.DataFrame(), np.array([]), [], []

    model, train_summary = train_model(train_contexts, train_targets, val_contexts, val_targets, args)
    device = torch.device(str(train_summary["device"]))
    threshold_source = "window_validation"
    threshold_mode = "per_window_validation"
    calibration_scores = np.array([])
    threshold = float("nan")
    quantile_threshold = float("nan")

    if args.threshold_calibration_split == "window_validation":
        calibration_scores = score_samples(model, val_contexts, val_targets, args, device)
        threshold, quantile_threshold = compute_threshold_from_scores(calibration_scores, args)
    else:
        if calibration_contexts is None or calibration_targets is None:
            raise ValueError("External threshold calibration samples were not prepared.")
        threshold_source = str(args.threshold_calibration_split)
        threshold_mode = "global_pooled"
        calibration_scores = score_samples(
            model,
            calibration_contexts,
            calibration_targets,
            args,
            device,
        )

    test_raw_scores = anomaly_scores(model, test_contexts, test_targets, device, batch_size=args.score_batch_size)
    test_scores = smooth_scores(test_raw_scores, args.smooth_window)
    scores_df = test_meta_df.copy()
    scores_df["window_step"] = window_step
    scores_df["train_start_s"] = window["train_start_s"]
    scores_df["train_end_s"] = window["train_end_s"]
    scores_df["test_start_s"] = window["test_start_s"]
    scores_df["test_end_s"] = window["test_end_s"]
    scores_df["raw_anomaly_score"] = test_raw_scores
    scores_df["anomaly_score"] = test_scores

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
        "threshold_calibration_mode": threshold_mode,
        "threshold": threshold,
        "threshold_calibration_split": threshold_source,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1_score": float("nan"),
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "roc_auc": float("nan"),
        "epochs_run": int(train_summary["epochs_run"]),
        "best_epoch": int(train_summary["best_epoch"]),
        "best_val_loss": float(train_summary["best_val_loss"]),
        "quantile_threshold": float(quantile_threshold),
        "calib_score_mean": float(np.mean(calibration_scores)) if len(calibration_scores) > 0 else float("nan"),
        "calib_score_std": float(np.std(calibration_scores)) if len(calibration_scores) > 0 else float("nan"),
        "calib_score_p50": float(np.median(calibration_scores)) if len(calibration_scores) > 0 else float("nan"),
        "calib_score_p99": float(np.quantile(calibration_scores, 0.99)) if len(calibration_scores) > 0 else float("nan"),
    }

    if args.threshold_calibration_split == "window_validation":
        test_labels = test_meta_df["is_attack"].astype(int).to_numpy()
        test_pred, metric_values = compute_metrics_at_threshold(test_labels, test_scores, threshold)
        scores_df["pred_is_anomaly"] = test_pred
        scores_df["threshold"] = threshold
        metrics_row.update(metric_values)

    return metrics_row, scores_df, calibration_scores, list(train_summary["train_losses"]), list(train_summary["val_losses"])


def prepare_sample_arrays(
    df: pd.DataFrame,
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    contexts, targets, _ = build_packet_samples(df, args.context_length, token_to_idx)
    return contexts, targets


def trusted_update_mask(
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.trusted_update_score_source == "raw":
        return raw_scores < threshold
    return smoothed_scores < threshold


def run_trusted_online_update(
    df: pd.DataFrame,
    windows: list[dict[str, float]],
    token_to_idx: dict[str, int],
    threshold_contexts: np.ndarray,
    threshold_targets: np.ndarray,
    update_val_contexts: np.ndarray,
    update_val_targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float], list[float]]:
    warmup_window = windows[0]
    warmup_df = slice_by_time(df, warmup_window["train_start_s"], warmup_window["train_end_s"])
    warmup_cutoff_s = warmup_window["train_start_s"] + (
        warmup_window["train_end_s"] - warmup_window["train_start_s"]
    ) * (1.0 - args.val_fraction)
    warmup_train_df = warmup_df[warmup_df["time_offset_s"] < warmup_cutoff_s].copy()

    warmup_train_contexts, warmup_train_targets = prepare_sample_arrays(warmup_train_df, token_to_idx, args)
    if len(warmup_train_contexts) == 0:
        raise ValueError("Warmup window does not produce enough training samples.")

    model = create_model(args)
    model, warmup_summary = fit_model(
        model,
        warmup_train_contexts,
        warmup_train_targets,
        update_val_contexts,
        update_val_targets,
        args.epochs,
        args.patience,
        args,
    )
    last_train_losses = list(warmup_summary["train_losses"])
    last_val_losses = list(warmup_summary["val_losses"])

    seen_until_s = float(warmup_window["train_end_s"])
    metrics_rows: list[dict[str, Any]] = []
    all_scores: list[pd.DataFrame] = []

    for window_step, window in enumerate(windows, start=1):
        device = torch.device("cpu")
        calibration_scores = score_samples(model, threshold_contexts, threshold_targets, args, device)
        threshold, quantile_threshold = compute_threshold_from_scores(calibration_scores, args)

        test_contexts, test_targets, test_meta_df = build_test_samples_with_context(
            df,
            window["test_start_s"],
            window["test_end_s"],
            args.context_length,
            token_to_idx,
        )
        if len(test_contexts) == 0:
            metrics_rows.append(
                {
                    "window_step": window_step,
                    **window,
                    "status": "skipped",
                    "skip_reason": "insufficient_test_samples",
                    "threshold_calibration_mode": "online_fixed_split",
                    "threshold_calibration_split": args.threshold_calibration_split,
                }
            )
            continue

        test_raw_scores = anomaly_scores(model, test_contexts, test_targets, device, batch_size=args.score_batch_size)
        test_scores = smooth_scores(test_raw_scores, args.smooth_window)
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
        all_scores.append(scores_df)

        new_region_mask = (test_meta_df["time_offset_s"].to_numpy(dtype=float) >= seen_until_s)
        trusted_mask = new_region_mask & trusted_update_mask(
            test_raw_scores,
            test_scores,
            threshold,
            args,
        )
        trusted_contexts = test_contexts[trusted_mask]
        trusted_targets = test_targets[trusted_mask]

        update_summary: dict[str, Any] = {
            "epochs_run": 0,
            "best_epoch": 0,
            "best_val_loss": float("nan"),
        }
        if len(trusted_contexts) >= args.min_trusted_update_samples:
            model, update_summary = fit_model(
                model,
                trusted_contexts,
                trusted_targets,
                update_val_contexts,
                update_val_targets,
                args.online_update_epochs,
                args.online_update_patience,
                args,
            )
            last_train_losses = list(update_summary["train_losses"])
            last_val_losses = list(update_summary["val_losses"])

        metrics_rows.append(
            {
                "window_step": window_step,
                **window,
                "status": "ok",
                "skip_reason": "",
                "train_packets": int(len(warmup_train_df)) if window_step == 1 else int(len(trusted_contexts)),
                "val_packets": int(len(update_val_targets)),
                "test_packets": int(len(test_meta_df)),
                "train_samples": int(len(warmup_train_contexts)) if window_step == 1 else int(len(trusted_contexts)),
                "val_samples": int(len(update_val_targets)),
                "test_samples": int(len(test_contexts)),
                "train_attack_frac": 0.0,
                "test_attack_frac": float(test_meta_df["is_attack"].mean()) if len(test_meta_df) else 0.0,
                "threshold_calibration_mode": "online_fixed_split",
                "threshold_calibration_split": args.threshold_calibration_split,
                "threshold": threshold,
                "quantile_threshold": quantile_threshold,
                "calib_score_mean": float(np.mean(calibration_scores)),
                "calib_score_std": float(np.std(calibration_scores)),
                "calib_score_p50": float(np.median(calibration_scores)),
                "calib_score_p99": float(np.quantile(calibration_scores, 0.99)),
                "trusted_new_samples": int(new_region_mask.sum()),
                "trusted_update_samples": int(len(trusted_contexts)),
                "trusted_update_fraction": (
                    float(len(trusted_contexts) / max(1, int(new_region_mask.sum())))
                ),
                "trusted_update_score_source": args.trusted_update_score_source,
                "epochs_run": int(update_summary["epochs_run"]),
                "best_epoch": int(update_summary["best_epoch"]),
                "best_val_loss": float(update_summary["best_val_loss"]),
                **metric_values,
            }
        )
        seen_until_s = max(seen_until_s, float(window["test_end_s"]))

    metrics_df = pd.DataFrame(metrics_rows)
    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    return metrics_df, scores_df, last_train_losses, last_val_losses


def plot_score_timeline(
    scores_df: pd.DataFrame,
    attack_segments: list[tuple[float, float, str]],
    output_path: Path,
) -> None:
    if scores_df.empty:
        return

    aggregated = (
        scores_df.groupby("time_offset_s", as_index=False)
        .agg(
            anomaly_score=("anomaly_score", "mean"),
            threshold=("threshold", "mean"),
        )
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
        default="artifacts/rolling_fc_address/rolling_packet_features_fc_address.csv",
        help="Packet-level rolling CSV generated by pre_process_rolling_fc_address.py",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address/sliding_window_results",
        help="Directory for sliding-window packet-level LSTM outputs.",
    )
    parser.add_argument("--pcap-name", default="mixed_long_conti")
    parser.add_argument("--train-duration-s", type=float, default=600.0)
    parser.add_argument("--test-duration-s", type=float, default=300.0)
    parser.add_argument("--step-s", type=float, default=120.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--self-clean-rounds", type=int, default=0)
    parser.add_argument(
        "--self-clean-quantile",
        type=float,
        default=0.95,
        help="During self-cleaning, drop target packets whose anomaly score is at or above this score quantile.",
    )
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
    parser.add_argument("--validation-quantile", type=float, default=0.99)
    parser.add_argument(
        "--threshold-method",
        default="sigma",
        choices=["quantile", "sigma"],
        help="'sigma' sets threshold = mean + k*std of calibration scores (robust when normal scores cluster near zero); "
             "'quantile' uses the validation_quantile percentile (original behaviour).",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Number of standard deviations above the calibration mean to use as threshold (--threshold-method sigma only).",
    )
    parser.add_argument(
        "--threshold-calibration-split",
        default="window_validation",
        help="Use 'window_validation' for per-window tail calibration or a dataset split name like 'validation' for pure-normal calibration.",
    )
    parser.add_argument(
        "--learning-mode",
        default="window_retrain",
        choices=["window_retrain", "trusted_online_update"],
        help="window_retrain trains one fresh model per window; trusted_online_update trains one warmup model and only updates on trusted low-score samples.",
    )
    parser.add_argument(
        "--online-update-epochs",
        type=int,
        default=5,
        help="Epochs for each trusted online update step.",
    )
    parser.add_argument(
        "--online-update-patience",
        type=int,
        default=2,
        help="Early-stopping patience for each trusted online update step.",
    )
    parser.add_argument(
        "--trusted-update-score-source",
        default="raw",
        choices=["raw", "smoothed"],
        help="Use raw or smoothed anomaly scores to decide whether a new sample is trusted enough for online update.",
    )
    parser.add_argument(
        "--min-trusted-update-samples",
        type=int,
        default=32,
        help="Minimum trusted samples required before applying an online update step.",
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

    df = df.sort_values("timestamp").reset_index(drop=True)
    token_source_df = df
    calibration_contexts: np.ndarray | None = None
    calibration_targets: np.ndarray | None = None

    if args.threshold_calibration_split != "window_validation":
        threshold_df = full_df[full_df["split"] == args.threshold_calibration_split].copy()
        if threshold_df.empty:
            raise ValueError(
                f"No rows found for threshold calibration split={args.threshold_calibration_split}"
            )
        token_source_df = pd.concat([df, threshold_df], ignore_index=True)

    token_to_idx = build_token_mapping(token_source_df)
    args.vocab_size = len(token_to_idx)

    if args.threshold_calibration_split != "window_validation":
        threshold_df = full_df[full_df["split"] == args.threshold_calibration_split].copy()
        threshold_df = threshold_df[threshold_df["pair_token"].isin(token_to_idx)].copy()
        calibration_contexts, calibration_targets, _ = build_packet_samples(
            threshold_df,
            args.context_length,
            token_to_idx,
        )
        if len(calibration_contexts) == 0:
            raise ValueError(
                f"Threshold calibration split={args.threshold_calibration_split} does not produce usable samples."
            )

    windows = build_time_windows(df, args.train_duration_s, args.test_duration_s, args.step_s)
    if not windows:
        raise ValueError("No sliding windows fit within the selected PCAP timeline.")

    global_threshold = float("nan")
    global_quantile_threshold = float("nan")
    global_calibration_score_count = 0
    last_train_losses: list[float] = []
    last_val_losses: list[float] = []

    if args.learning_mode == "trusted_online_update":
        if calibration_contexts is None or calibration_targets is None:
            raise ValueError(
                "trusted_online_update requires --threshold-calibration-split to reference a clean split such as 'validation'."
            )
        metrics_df, scores_df, last_train_losses, last_val_losses = run_trusted_online_update(
            df,
            windows,
            token_to_idx,
            calibration_contexts,
            calibration_targets,
            calibration_contexts,
            calibration_targets,
            args,
        )
        calibration_model = create_model(args)
        warmup_window = windows[0]
        warmup_df = slice_by_time(df, warmup_window["train_start_s"], warmup_window["train_end_s"])
        warmup_cutoff_s = warmup_window["train_start_s"] + (
            warmup_window["train_end_s"] - warmup_window["train_start_s"]
        ) * (1.0 - args.val_fraction)
        warmup_train_df = warmup_df[warmup_df["time_offset_s"] < warmup_cutoff_s].copy()
        warmup_train_contexts, warmup_train_targets = prepare_sample_arrays(warmup_train_df, token_to_idx, args)
        calibration_model, _ = fit_model(
            calibration_model,
            warmup_train_contexts,
            warmup_train_targets,
            calibration_contexts,
            calibration_targets,
            args.epochs,
            args.patience,
            args,
        )
        pooled_calibration_scores = score_samples(
            calibration_model,
            calibration_contexts,
            calibration_targets,
            args,
            torch.device("cpu"),
        )
        global_calibration_score_count = int(len(pooled_calibration_scores))
        global_threshold, global_quantile_threshold = compute_threshold_from_scores(pooled_calibration_scores, args)
    else:
        metrics_rows: list[dict[str, Any]] = []
        all_scores: list[pd.DataFrame] = []
        all_calibration_scores: list[np.ndarray] = []

        for window_step, window in enumerate(windows, start=1):
            metrics_row, scores_df, calibration_scores, train_losses, val_losses = run_one_window(
                df,
                window_step,
                window,
                token_to_idx,
                calibration_contexts,
                calibration_targets,
                args,
            )
            metrics_rows.append(metrics_row)
            if not scores_df.empty:
                all_scores.append(scores_df)
            if len(calibration_scores) > 0:
                all_calibration_scores.append(calibration_scores)
            if train_losses and val_losses:
                last_train_losses = train_losses
                last_val_losses = val_losses

        metrics_df = pd.DataFrame(metrics_rows)
        scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()

        if args.threshold_calibration_split != "window_validation":
            if not all_calibration_scores:
                raise ValueError("No calibration scores were collected for global threshold calibration.")
            pooled_calibration_scores = np.concatenate(all_calibration_scores)
            global_calibration_score_count = int(len(pooled_calibration_scores))
            global_threshold, global_quantile_threshold = compute_threshold_from_scores(
                pooled_calibration_scores,
                args,
            )

            if not scores_df.empty:
                scores_df["threshold"] = global_threshold
                scores_df["pred_is_anomaly"] = (
                    scores_df["anomaly_score"].to_numpy(dtype=float) >= global_threshold
                ).astype(int)

            for idx, row in metrics_df.iterrows():
                if row["status"] != "ok":
                    continue
                window_scores_df = scores_df[scores_df["window_step"] == row["window_step"]].copy()
                if window_scores_df.empty:
                    continue
                test_labels = window_scores_df["is_attack"].astype(int).to_numpy()
                test_scores = window_scores_df["anomaly_score"].to_numpy(dtype=float)
                _, metric_values = compute_metrics_at_threshold(test_labels, test_scores, global_threshold)
                metrics_df.loc[idx, "threshold"] = global_threshold
                metrics_df.loc[idx, "quantile_threshold"] = global_quantile_threshold
                for metric_name, metric_value in metric_values.items():
                    metrics_df.loc[idx, metric_name] = metric_value

    metrics_path = output_dir / "sliding_window_step_metrics.csv"
    scores_path = output_dir / "sliding_window_all_scores.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not scores_df.empty:
        scores_df.to_csv(scores_path, index=False)

    successful_metrics_df = metrics_df[metrics_df["status"] == "ok"].copy()
    attack_segments = contiguous_attack_segments(df)
    plot_score_timeline(scores_df, attack_segments, output_dir / "sliding_window_score_timeline.png")
    plot_metrics(successful_metrics_df, output_dir / "sliding_window_metrics_over_time.png")
    plot_thresholds(successful_metrics_df, output_dir / "sliding_window_threshold_over_time.png")

    if last_train_losses and last_val_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(last_train_losses, label="train_loss")
        plt.plot(last_val_losses, label="val_cross_entropy")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Sliding Window Last Training Curve")
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
        "val_fraction": args.val_fraction,
        "self_clean_rounds": args.self_clean_rounds,
        "self_clean_quantile": args.self_clean_quantile,
        "context_length": args.context_length,
        "smooth_window": args.smooth_window,
        "validation_quantile": args.validation_quantile,
        "threshold_calibration_split": args.threshold_calibration_split,
        "threshold_calibration_mode": (
            "per_window_validation"
            if args.learning_mode == "window_retrain" and args.threshold_calibration_split == "window_validation"
            else (
                "global_pooled"
                if args.learning_mode == "window_retrain"
                else "online_fixed_split"
            )
        ),
        "learning_mode": args.learning_mode,
        "online_update_epochs": args.online_update_epochs,
        "online_update_patience": args.online_update_patience,
        "trusted_update_score_source": args.trusted_update_score_source,
        "min_trusted_update_samples": args.min_trusted_update_samples,
        "global_threshold": global_threshold,
        "global_quantile_threshold": global_quantile_threshold,
        "global_calibration_score_count": global_calibration_score_count,
        "vocab_size": len(token_to_idx),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
