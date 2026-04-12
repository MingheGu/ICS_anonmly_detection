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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class PacketRollLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 8,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.output(last_hidden)


def build_token_mapping(df: pd.DataFrame) -> dict[str, int]:
    tokens = sorted(df["pair_token"].astype(str).unique().tolist())
    return {token: idx for idx, token in enumerate(tokens)}


def build_packet_samples(
    df: pd.DataFrame,
    context_length: int,
    token_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, object]] = []

    for pcap_name, pcap_df in df.groupby("pcap_name"):
        pcap_df = pcap_df.sort_values("timestamp").reset_index(drop=True)
        token_indices = pcap_df["pair_token"].map(token_to_idx).to_numpy(dtype=np.int64)

        if len(pcap_df) <= context_length:
            continue

        for idx in range(context_length, len(pcap_df)):
            contexts.append(token_indices[idx - context_length:idx])
            targets.append(int(token_indices[idx]))
            metadata.append(
                {
                    "pcap_name": str(pcap_name),
                    "timestamp": float(pcap_df.loc[idx, "timestamp"]),
                    "time_offset_s": float(pcap_df.loc[idx, "time_offset_s"]),
                    "label": str(pcap_df.loc[idx, "label"]),
                    "is_attack": int(pcap_df.loc[idx, "is_attack"]),
                    "split": str(pcap_df.loc[idx, "split"]),
                    "pair_token": str(pcap_df.loc[idx, "pair_token"]),
                    "src_ip": str(pcap_df.loc[idx, "src_ip"]),
                    "dst_ip": str(pcap_df.loc[idx, "dst_ip"]),
                    "dst_port": int(pcap_df.loc[idx, "dst_port"]),
                    "function_code": int(pcap_df.loc[idx, "function_code"]),
                    "address": int(pcap_df.loc[idx, "address"]),
                }
            )

    return np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64), metadata


def cross_entropy_loss(
    model: nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    if len(contexts) == 0:
        return float("nan")
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            end = start + batch_size
            x = torch.tensor(contexts[start:end], dtype=torch.long, device=device)
            y = torch.tensor(targets[start:end], dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def anomaly_scores(
    model: nn.Module,
    contexts: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            end = start + batch_size
            x = torch.tensor(contexts[start:end], dtype=torch.long, device=device)
            y = torch.tensor(targets[start:end], dtype=torch.long, device=device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            true_probs = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            scores = -torch.log(true_probs + 1e-12)
            all_scores.append(scores.detach().cpu().numpy())
    return np.concatenate(all_scores, axis=0) if all_scores else np.array([])


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(scores) == 0:
        return scores.astype(float, copy=True)
    return (
        pd.Series(scores, dtype=float)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-dataset",
        default="artifacts/rolling_fc_address/rolling_packet_features_fc_address.csv",
        help="Packet-level rolling CSV generated by pre_process_rolling_fc_address.py",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address/packet_results_v3",
        help="Directory for packet-level rolling LSTM outputs.",
    )
    parser.add_argument("--context-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-quantile",
        type=float,
        default=0.99,
        help="Quantile of validation anomaly score used as threshold.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / args.packet_dataset
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    token_to_idx = build_token_mapping(df)
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    contexts, targets, metadata = build_packet_samples(df, args.context_length, token_to_idx)
    metadata_df = pd.DataFrame(metadata)

    train_mask = metadata_df["split"] == "train"
    val_mask = metadata_df["split"] == "validation"
    holdout_mask = metadata_df["split"] == "normal_holdout"
    test_mask = metadata_df["split"] == "test"

    train_contexts = contexts[train_mask.to_numpy()]
    train_targets = targets[train_mask.to_numpy()]
    val_contexts = contexts[val_mask.to_numpy()]
    val_targets = targets[val_mask.to_numpy()]
    holdout_contexts = contexts[holdout_mask.to_numpy()]
    holdout_targets = targets[holdout_mask.to_numpy()]
    test_contexts = contexts[test_mask.to_numpy()]
    test_targets = targets[test_mask.to_numpy()]
    test_labels = metadata_df.loc[test_mask, "is_attack"].astype(int).to_numpy()

    if len(train_contexts) == 0 or len(val_contexts) == 0 or len(test_contexts) == 0:
        raise ValueError("One of train/validation/test packet sample sets is empty.")

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

    train_scores = anomaly_scores(model, train_contexts, train_targets, device)
    val_raw_scores = anomaly_scores(model, val_contexts, val_targets, device)
    holdout_raw_scores = anomaly_scores(model, holdout_contexts, holdout_targets, device) if len(holdout_contexts) else np.array([])
    test_raw_scores = anomaly_scores(model, test_contexts, test_targets, device)

    val_scores = val_raw_scores
    holdout_scores = holdout_raw_scores if len(holdout_raw_scores) else np.array([])
    test_scores = test_raw_scores

    smoothed_val_max = float(np.max(val_scores))
    quantile_threshold = float(np.quantile(val_scores, args.validation_quantile))
    threshold = float(max(smoothed_val_max, 1e-6))
    test_pred = (test_scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        test_pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred, labels=[0, 1]).ravel()

    metrics_df = pd.DataFrame(
        [
            {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "threshold": threshold,
                "smoothed_val_max": smoothed_val_max,
                "quantile_threshold": quantile_threshold,
                "validation_quantile": args.validation_quantile,
                "smooth_window": 1,
                "context_length": args.context_length,
                "epochs_requested": args.epochs,
                "epochs_run": int(len(train_losses)),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_val_loss),
                "patience": args.patience,
                "dropout": args.dropout,
                "vocab_size": len(token_to_idx),
                "train_rows": int(len(train_contexts)),
                "validation_rows": int(len(val_contexts)),
                "normal_holdout_rows": int(len(holdout_contexts)),
                "test_rows": int(len(test_contexts)),
            }
        ]
    )
    metrics_df.to_csv(output_dir / "packet_roll_lstm_metrics.csv", index=False)

    cm = confusion_matrix(test_labels, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["normal", "anomaly"], columns=["pred_normal", "pred_anomaly"])
    cm_df.to_csv(output_dir / "packet_roll_lstm_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Packet-level Rolling LSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "packet_roll_lstm_confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Packet-level Rolling LSTM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "packet_roll_lstm_roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_cross_entropy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Packet-level Rolling LSTM Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "packet_roll_lstm_training_curve.png", dpi=200)
    plt.close()

    test_scores_df = metadata_df.loc[test_mask].reset_index(drop=True).copy()
    test_scores_df["raw_anomaly_score"] = test_raw_scores
    test_scores_df["anomaly_score"] = test_scores
    test_scores_df["pred_is_anomaly"] = test_pred
    test_scores_df["target_token_index"] = test_targets
    test_scores_df["target_token"] = [idx_to_token[int(idx)] for idx in test_targets]
    test_scores_df.to_csv(output_dir / "packet_roll_test_scores.csv", index=False)

    val_scores_df = metadata_df.loc[val_mask].reset_index(drop=True).copy()
    val_scores_df["raw_anomaly_score"] = val_raw_scores
    val_scores_df["anomaly_score"] = val_scores
    val_scores_df["target_token_index"] = val_targets
    val_scores_df["target_token"] = [idx_to_token[int(idx)] for idx in val_targets]
    val_scores_df.to_csv(output_dir / "packet_roll_validation_scores.csv", index=False)

    if len(holdout_contexts):
        holdout_scores_df = metadata_df.loc[holdout_mask].reset_index(drop=True).copy()
        holdout_scores_df["raw_anomaly_score"] = holdout_raw_scores
        holdout_scores_df["anomaly_score"] = holdout_scores
        holdout_scores_df["pred_is_anomaly"] = (holdout_scores >= threshold).astype(int)
        holdout_scores_df["target_token_index"] = holdout_targets
        holdout_scores_df["target_token"] = [idx_to_token[int(idx)] for idx in holdout_targets]
        holdout_scores_df.to_csv(output_dir / "packet_roll_holdout_scores.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(token_to_idx),
            "embed_dim": args.embed_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "context_length": args.context_length,
            "threshold": threshold,
            "smoothed_val_max": smoothed_val_max,
            "quantile_threshold": quantile_threshold,
            "smooth_window": 1,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "token_to_idx": token_to_idx,
        },
        output_dir / "packet_roll_lstm_model.pt",
    )

    run_summary = {
        "packet_dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "context_length": args.context_length,
        "epochs_requested": args.epochs,
        "epochs_run": int(len(train_losses)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "patience": args.patience,
        "dropout": args.dropout,
        "vocab_size": len(token_to_idx),
        "threshold": threshold,
        "smoothed_val_max": smoothed_val_max,
        "quantile_threshold": quantile_threshold,
        "validation_quantile": args.validation_quantile,
        "smooth_window": 1,
        "train_rows": int(len(train_contexts)),
        "validation_rows": int(len(val_contexts)),
        "normal_holdout_rows": int(len(holdout_contexts)),
        "test_rows": int(len(test_contexts)),
        "roc_auc": roc_auc,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
