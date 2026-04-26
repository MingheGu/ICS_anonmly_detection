from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch

from transformer_window_common import (
    anomaly_scores,
    build_packet_samples,
    build_test_samples_with_context,
    compute_metrics_at_threshold,
    compute_threshold_from_scores,
    create_model,
    fit_model,
    score_samples,
    smooth_scores,
)
from lstm_window_common import slice_by_time


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
