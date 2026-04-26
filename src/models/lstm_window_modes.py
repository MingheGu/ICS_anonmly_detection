from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch

from lstm_window_common import (
    build_packet_samples,
    build_static_window,
    build_test_samples_with_context,
    compute_metrics_at_threshold,
    compute_threshold_from_scores,
    create_model,
    fit_model,
    remove_high_score_targets,
    score_samples,
    slice_by_time,
    smooth_scores,
    anomaly_scores,
)


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


def run_static_mode(
    df: pd.DataFrame,
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], list[dict[str, Any]], pd.DataFrame, list[float], list[float]]:
    window = build_static_window(df, args.train_duration_s, args.test_duration_s)
    metrics_row, scores_df, train_losses, val_losses = run_one_window(
        df,
        1,
        window,
        token_to_idx,
        args,
    )
    return [window], [metrics_row], scores_df, train_losses, val_losses


def run_fixed_train_mode(
    df: pd.DataFrame,
    windows: list[dict[str, float]],
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[float], list[float]]:
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
            df, window["test_start_s"], window["test_end_s"], args.context_length, token_to_idx
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


def run_sliding_retrain_mode(
    df: pd.DataFrame,
    windows: list[dict[str, float]],
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[float], list[float]]:
    metrics_rows: list[dict[str, Any]] = []
    all_scores: list[pd.DataFrame] = []
    last_train_losses: list[float] = []
    last_val_losses: list[float] = []

    for window_step, window in enumerate(windows, start=1):
        metrics_row, scores_df_w, train_losses, val_losses = run_one_window(
            df, window_step, window, token_to_idx, args
        )
        metrics_rows.append(metrics_row)
        if not scores_df_w.empty:
            all_scores.append(scores_df_w)
        if train_losses and val_losses:
            last_train_losses = train_losses
            last_val_losses = val_losses

    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    return metrics_rows, scores_df, last_train_losses, last_val_losses
