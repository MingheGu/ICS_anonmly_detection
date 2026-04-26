from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_roll_packet_lstm_v2 import (
    PacketRollLSTM,
    anomaly_scores,
    build_packet_samples,
    build_token_mapping,
    cross_entropy_loss,
    set_seed,
)


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(float, copy=True)
    return (
        pd.Series(scores, dtype=float)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def train_one_pass(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, object]]:
    train_contexts, train_targets, _ = build_packet_samples(train_df, args.context_length, token_to_idx)
    val_contexts, val_targets, _ = build_packet_samples(val_df, args.context_length, token_to_idx)

    if len(train_contexts) == 0 or len(val_contexts) == 0:
        raise ValueError("One of train/validation packet sample sets is empty for this expanding pass.")

    device = torch.device("cpu")
    model = PacketRollLSTM(
        vocab_size=len(token_to_idx),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
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

    val_raw_scores = anomaly_scores(model, val_contexts, val_targets, device)
    val_scores = smooth_scores(val_raw_scores, args.smooth_window)
    quantile_threshold = float(np.quantile(val_scores, args.validation_quantile))
    threshold = float(max(quantile_threshold, 1e-6))

    summary: dict[str, object] = {
        "device": str(device),
        "train_rows": int(len(train_contexts)),
        "validation_rows": int(len(val_contexts)),
        "epochs_requested": int(args.epochs),
        "epochs_run": int(len(train_losses)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "threshold": threshold,
        "quantile_threshold": quantile_threshold,
        "validation_quantile": float(args.validation_quantile),
        "val_raw_scores": val_raw_scores,
        "val_scores": val_scores,
    }
    return model, summary


def evaluate_pass(
    model: nn.Module,
    test_df: pd.DataFrame,
    token_to_idx: dict[str, int],
    args: argparse.Namespace,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    test_contexts, test_targets, metadata = build_packet_samples(test_df, args.context_length, token_to_idx)
    if len(test_contexts) == 0:
        raise ValueError("Test packet sample set is empty for this expanding pass.")

    metadata_df = pd.DataFrame(metadata)
    labels = metadata_df["is_attack"].astype(int).to_numpy()
    device = torch.device("cpu")
    raw_scores = anomaly_scores(model, test_contexts, test_targets, device)
    scores = smooth_scores(raw_scores, args.smooth_window)
    pred = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    scores_df = metadata_df.copy()
    scores_df["raw_anomaly_score"] = raw_scores
    scores_df["anomaly_score"] = scores
    scores_df["pred_is_anomaly"] = pred
    scores_df["target_token_index"] = test_targets

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "roc_auc": roc_auc,
        "test_rows": int(len(test_contexts)),
    }
    return scores_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-dataset",
        default="artifacts/rolling_fc_address/rolling_packet_features_fc_address.csv",
        help="Packet-level rolling CSV generated by pre_process_rolling_fc_address.py",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address/packet_expand_results",
        help="Directory for expanding-window packet-level LSTM outputs.",
    )
    parser.add_argument("--context-length", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-quantile", type=float, default=0.99)
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / args.packet_dataset
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    token_to_idx = build_token_mapping(df)

    base_train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_pcaps = df.loc[df["split"] == "test", "pcap_name"].drop_duplicates().tolist()

    if base_train_df.empty or val_df.empty or not test_pcaps:
        raise ValueError("Train/validation/test splits are incomplete for expanding-window training.")

    expanded_train_df = base_train_df.copy()
    all_scores: list[pd.DataFrame] = []
    pass_rows: list[dict[str, object]] = []

    for pass_index, pcap_name in enumerate(test_pcaps, start=1):
        test_df = df[df["pcap_name"] == pcap_name].copy()
        model, train_summary = train_one_pass(expanded_train_df, val_df, token_to_idx, args)
        threshold = float(train_summary["threshold"])

        scores_df, metrics = evaluate_pass(model, test_df, token_to_idx, args, threshold)
        scores_df["pass_index"] = pass_index
        all_scores.append(scores_df)

        pass_rows.append(
            {
                "pass_index": pass_index,
                "test_pcap_name": pcap_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "roc_auc": metrics["roc_auc"],
                "threshold": threshold,
                "quantile_threshold": float(train_summary["quantile_threshold"]),
                "validation_quantile": float(train_summary["validation_quantile"]),
                "smooth_window": int(args.smooth_window),
                "train_rows_before_expand": int(train_summary["train_rows"]),
                "validation_rows": int(train_summary["validation_rows"]),
                "test_rows": int(metrics["test_rows"]),
                "epochs_run": int(train_summary["epochs_run"]),
                "best_epoch": int(train_summary["best_epoch"]),
                "best_validation_loss": float(train_summary["best_validation_loss"]),
            }
        )

        newly_revealed_normal = test_df[test_df["is_attack"] == 0].copy()
        if not newly_revealed_normal.empty:
            expanded_train_df = pd.concat([expanded_train_df, newly_revealed_normal], ignore_index=True)

    pass_metrics_df = pd.DataFrame(pass_rows)
    pass_metrics_df.to_csv(output_dir / "expanding_pass_metrics.csv", index=False)

    all_scores_df = pd.concat(all_scores, ignore_index=True)
    all_scores_df.to_csv(output_dir / "expanding_all_test_scores.csv", index=False)

    overall_labels = all_scores_df["is_attack"].astype(int).to_numpy()
    overall_scores = all_scores_df["anomaly_score"].to_numpy(dtype=float)
    overall_pred = all_scores_df["pred_is_anomaly"].astype(int).to_numpy()
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        overall_labels,
        overall_pred,
        average="binary",
        zero_division=0,
    )
    overall_tn, overall_fp, overall_fn, overall_tp = confusion_matrix(
        overall_labels,
        overall_pred,
        labels=[0, 1],
    ).ravel()
    overall_fpr, overall_tpr, _ = roc_curve(overall_labels, overall_scores)
    overall_auc = float(auc(overall_fpr, overall_tpr))

    plt.figure(figsize=(6, 4.5))
    plt.plot(overall_fpr, overall_tpr, label=f"AUC={overall_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Expanding-window Packet LSTM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "expanding_roc_curve.png", dpi=200)
    plt.close()

    cm = confusion_matrix(overall_labels, overall_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["normal", "anomaly"], columns=["pred_normal", "pred_anomaly"])
    cm_df.to_csv(output_dir / "expanding_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Expanding-window Packet LSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "expanding_confusion_matrix.png", dpi=200)
    plt.close()

    summary = {
        "packet_dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "context_length": args.context_length,
        "smooth_window": int(args.smooth_window),
        "validation_quantile": args.validation_quantile,
        "num_passes": int(len(test_pcaps)),
        "test_pcaps": test_pcaps,
        "initial_train_packet_rows": int(len(base_train_df)),
        "validation_packet_rows": int(len(val_df)),
        "final_expanded_train_packet_rows": int(len(expanded_train_df)),
        "overall_precision": float(overall_precision),
        "overall_recall": float(overall_recall),
        "overall_f1_score": float(overall_f1),
        "overall_tp": int(overall_tp),
        "overall_fp": int(overall_fp),
        "overall_tn": int(overall_tn),
        "overall_fn": int(overall_fn),
        "overall_roc_auc": overall_auc,
    }
    (output_dir / "expanding_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(pass_metrics_df.to_string(index=False))
    print(
        f"\nOverall precision={overall_precision:.4f} "
        f"recall={overall_recall:.4f} f1={overall_f1:.4f} auc={overall_auc:.4f}"
    )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
