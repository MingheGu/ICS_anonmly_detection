from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16, latent_size: int = 8) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def scale_sequences(
    train_seq: np.ndarray,
    other_sets: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    flat_train = train_seq.reshape(-1, train_seq.shape[-1])
    scaler.fit(flat_train)

    def transform(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, arr.shape[-1])
        scaled = scaler.transform(flat)
        return scaled.reshape(arr.shape).astype(np.float32)

    return transform(train_seq), [transform(arr) for arr in other_sets], scaler


def reconstruction_scores(model: nn.Module, sequences: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        reconstructed = model(tensor)
        mse = torch.mean((reconstructed - tensor) ** 2, dim=(1, 2))
    return mse.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        default="artifacts/lstm_fc_address",
        help="Directory containing lstm_sequences.npz and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/lstm_fc_address/results",
        help="Directory for model outputs.",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument(
        "--validation-quantile",
        type=float,
        default=0.97,
        help="Quantile of validation reconstruction error used as threshold.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sequence_dir = project_root / args.sequence_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(sequence_dir / "lstm_sequences.npz", allow_pickle=True)
    sequences = npz["sequences"].astype(np.float32)
    metadata = pd.read_csv(sequence_dir / "lstm_sequence_metadata.csv")

    train_mask = metadata["pcap_name"] == "normal_02"
    val_mask = metadata["pcap_name"] == "normal_03"
    test_mask = metadata["pcap_name"].isin(["attack_write_01", "attack_scan_02"])

    train_sequences = sequences[train_mask.to_numpy()]
    val_sequences = sequences[val_mask.to_numpy()]
    test_sequences = sequences[test_mask.to_numpy()]
    test_labels = (metadata.loc[test_mask, "label"] == "anomaly").astype(int).to_numpy()
    test_meta = metadata.loc[test_mask].reset_index(drop=True)

    train_sequences, [val_sequences, test_sequences], scaler = scale_sequences(
        train_sequences,
        [val_sequences, test_sequences],
    )

    device = torch.device("cpu")
    model = LSTMAutoencoder(
        input_size=train_sequences.shape[-1],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_sequences)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(args.epochs):
        model.train()
        batch_losses: list[float] = []
        for (batch,) in train_loader:
            batch = batch.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_losses.append(float(np.mean(batch_losses)))
        val_scores = reconstruction_scores(model, val_sequences, device)
        val_losses.append(float(np.mean(val_scores)))

    train_scores = reconstruction_scores(model, train_sequences, device)
    val_scores = reconstruction_scores(model, val_sequences, device)
    test_scores = reconstruction_scores(model, test_sequences, device)

    threshold = float(np.quantile(val_scores, args.validation_quantile))
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
                "validation_quantile": args.validation_quantile,
                "train_rows": int(len(train_sequences)),
                "validation_rows": int(len(val_sequences)),
                "test_rows": int(len(test_sequences)),
            }
        ]
    )
    metrics_df.to_csv(output_dir / "lstm_anomaly_metrics.csv", index=False)

    cm = confusion_matrix(test_labels, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["normal", "anomaly"], columns=["pred_normal", "pred_anomaly"])
    cm_df.to_csv(output_dir / "lstm_anomaly_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("LSTM Anomaly Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_anomaly_confusion_matrix.png", dpi=200)
    plt.close()

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM Anomaly ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_anomaly_roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_recon_error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_training_curve.png", dpi=200)
    plt.close()

    scores_df = test_meta.copy()
    scores_df["anomaly_score"] = test_scores
    scores_df["pred_is_anomaly"] = test_pred
    scores_df.to_csv(output_dir / "lstm_window_scores.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": int(train_sequences.shape[-1]),
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "sequence_length": int(train_sequences.shape[1]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "threshold": threshold,
        },
        output_dir / "lstm_autoencoder.pt",
    )

    run_summary = {
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "sequence_length": int(train_sequences.shape[1]),
        "feature_count": int(train_sequences.shape[2]),
        "epochs": args.epochs,
        "train_rows": int(len(train_sequences)),
        "validation_rows": int(len(val_sequences)),
        "test_rows": int(len(test_sequences)),
        "threshold": threshold,
        "validation_quantile": args.validation_quantile,
        "roc_auc": roc_auc,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
