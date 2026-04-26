from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def categorize_ratio(ratio: float) -> str:
    if ratio <= 0.20:
        return "weak_label_mixed"
    if ratio <= 0.50:
        return "partially_anomalous"
    return "strong_anomalous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        default="artifacts/lstm_fc_address/results/lstm_window_scores.csv",
        help="Path to the LSTM sequence scoring CSV.",
    )
    parser.add_argument(
        "--run-summary",
        default="artifacts/lstm_fc_address/results/run_summary.json",
        help="Path to the LSTM run summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/lstm_fc_address/results",
        help="Directory for the FN analysis outputs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scores_path = project_root / args.scores
    run_summary_path = project_root / args.run_summary
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.read_csv(scores_path)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    threshold = float(run_summary["threshold"])

    scores_df["window_label_list"] = scores_df["window_labels"].str.split("|")
    scores_df["sequence_window_count"] = scores_df["window_label_list"].str.len()
    scores_df["anomaly_window_count"] = scores_df["window_label_list"].apply(
        lambda labels: sum(label != "normal" for label in labels)
    )
    scores_df["anomaly_window_ratio"] = (
        scores_df["anomaly_window_count"] / scores_df["sequence_window_count"]
    )
    scores_df["fn_category"] = scores_df["anomaly_window_ratio"].apply(categorize_ratio)
    scores_df["score_margin_to_threshold"] = scores_df["anomaly_score"] - threshold

    fn_df = scores_df[
        (scores_df["label"] == "anomaly") & (scores_df["pred_is_anomaly"] == 0)
    ].copy()

    detailed_columns = [
        "pcap_name",
        "start_window_index",
        "end_window_index",
        "start_time_s",
        "end_time_s",
        "anomaly_score",
        "score_margin_to_threshold",
        "anomaly_window_count",
        "sequence_window_count",
        "anomaly_window_ratio",
        "fn_category",
        "window_labels",
    ]
    fn_df[detailed_columns].to_csv(output_dir / "lstm_false_negative_details.csv", index=False)

    summary = {
        "false_negative_count": int(len(fn_df)),
        "threshold": threshold,
        "by_pcap": fn_df["pcap_name"].value_counts().to_dict(),
        "by_category": fn_df["fn_category"].value_counts().to_dict(),
        "anomaly_window_ratio_stats": {
            "min": float(fn_df["anomaly_window_ratio"].min()) if not fn_df.empty else 0.0,
            "mean": float(fn_df["anomaly_window_ratio"].mean()) if not fn_df.empty else 0.0,
            "max": float(fn_df["anomaly_window_ratio"].max()) if not fn_df.empty else 0.0,
        },
        "score_margin_stats": {
            "min": float(fn_df["score_margin_to_threshold"].min()) if not fn_df.empty else 0.0,
            "mean": float(fn_df["score_margin_to_threshold"].mean()) if not fn_df.empty else 0.0,
            "max": float(fn_df["score_margin_to_threshold"].max()) if not fn_df.empty else 0.0,
        },
    }
    (output_dir / "lstm_false_negative_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    if not fn_df.empty:
        print("\nTop false negatives:")
        print(
            fn_df[detailed_columns]
            .sort_values(["anomaly_window_ratio", "anomaly_score"], ascending=[False, False])
            .head(15)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
