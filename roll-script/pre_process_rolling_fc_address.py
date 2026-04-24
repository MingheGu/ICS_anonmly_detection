from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from scapy.all import IP, PcapReader, Raw, TCP  # type: ignore


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def label_for_offset(offset_s: float, segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        start = float(segment["start"])
        end = segment["end"]
        if end is None:
            if offset_s >= start:
                return str(segment["label"])
            continue
        if start <= offset_s < float(end):
            return str(segment["label"])
    return str(segments[-1]["label"])


def extract_packet_features(packet: Any, target_ip: str) -> dict[str, Any] | None:
    if IP not in packet:
        return None

    ip_layer = packet[IP]
    if str(ip_layer.dst) != target_ip:
        return None

    if TCP not in packet:
        return None

    tcp_layer = packet[TCP]
    protocol = "TCP"
    src_port = int(tcp_layer.sport)
    dst_port = int(tcp_layer.dport)
    tcp_flags = str(tcp_layer.flags)
    payload = bytes(tcp_layer.payload)
    payload_len = len(payload)

    # Drop pure ACK noise, but keep SYN / FIN / SYN-ACK / payload-bearing packets.
    if payload_len == 0 and tcp_flags == "A":
        return None

    function_code = -1
    address = -1
    if dst_port == 502 and payload_len >= 10:
        protocol_id = int.from_bytes(payload[2:4], byteorder="big")
        mbap_length = int.from_bytes(payload[4:6], byteorder="big")
        candidate_fc = int(payload[7])
        if (
            protocol_id == 0
            and mbap_length == payload_len - 6
            and 1 <= candidate_fc <= 127
        ):
            function_code = candidate_fc
            address = int.from_bytes(payload[8:10], byteorder="big")

    return {
        "src_ip": str(ip_layer.src),
        "dst_ip": str(ip_layer.dst),
        "protocol": protocol,
        "src_port": src_port,
        "dst_port": dst_port,
        "tcp_flags": tcp_flags,
        "payload_len": payload_len,
        "function_code": function_code,
        "address": address,
    }


def refine_packet_label(
    coarse_label: str,
    features: dict[str, Any],
    attacker_ip: str,
    target_ip: str,
) -> str:
    if coarse_label == "normal":
        return "normal"

    src_ip = str(features["src_ip"])
    dst_ip = str(features["dst_ip"])

    if src_ip != attacker_ip or dst_ip != target_ip:
        return "normal"

    if coarse_label in {"attack_inject", "attack_inject_slow", "attack_inject_fuzz", "attack_write"}:
        if features["dst_port"] == 502 and features["function_code"] in {5, 6}:
            return "attack_inject"
        return "normal"  # TCP handshake, reads from attacker, etc.

    if coarse_label in {"attack_scan", "attack_scan_slow"}:
        return "attack_scan"

    return coarse_label


def process_pcap(
    project_root: Path,
    attacker_ip: str,
    target_ip: str,
    pcap_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pcap_path = project_root / str(pcap_cfg["path"])
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    first_timestamp: float | None = None

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            packet_ts = float(packet.time)
            if first_timestamp is None:
                first_timestamp = packet_ts

            features = extract_packet_features(packet, target_ip)
            if not features:
                continue

            offset_s = packet_ts - first_timestamp

            coarse_label = label_for_offset(offset_s, pcap_cfg["segments"])
            label = refine_packet_label(coarse_label, features, attacker_ip, target_ip)
            label_counts[label] += 1

            row = {
                "pcap_name": str(pcap_cfg["name"]),
                "pcap_path": str(pcap_cfg["path"]),
                "timestamp": packet_ts,
                "time_offset_s": round(offset_s, 6),
                "label": label,
                "is_attack": int(label != "normal"),
            }
            row.update(features)
            rows.append(row)

    return rows, label_counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows were extracted from the provided pcaps.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assign_split(row: pd.Series) -> str:
    if row["pcap_name"] == "normal_long_00":
        return "train"
    if row["pcap_name"] == "normal_long_03":
        return "validation"
    if row["pcap_name"] in {"mixed_long_03", "mixed_long_04"}:
        return "test"
    return "unused"


def assign_split_session(
    row: pd.Series,
    session_attack_onset_s: float | None,
    session_train_fraction: float,
) -> str:
    if row["pcap_name"] == "normal_long_00":
        return "train"
    if row["pcap_name"] == "normal_long_03":
        return "validation"
    if row["pcap_name"] in {"mixed_long_03", "mixed_long_04"}:
        return "test"
    if row["pcap_name"] == "session_ics" and session_attack_onset_s is not None:
        t = float(row["window_start_s"])
        train_cutoff = session_attack_onset_s * session_train_fraction
        if t < train_cutoff:
            return "train"
        if t < session_attack_onset_s:
            return "validation"
        return "test"
    return "unused"


def assign_packet_split(
    row: pd.Series,
    session_attack_onset_s: float | None = None,
    session_train_fraction: float = 0.8,
) -> str:
    if row["pcap_name"] == "normal_long_00":
        return "train"
    if row["pcap_name"] == "normal_long_03":
        return "validation"
    if row["pcap_name"] == "mixed_long_conti":
        return "sliding"
    if row["pcap_name"] in {"mixed_long_03", "mixed_long_04"}:
        return "test"
    if row["pcap_name"] == "session_ics" and session_attack_onset_s is not None:
        t = float(row["time_offset_s"])
        train_cutoff = session_attack_onset_s * session_train_fraction
        if t < train_cutoff:
            return "train"
        if t < session_attack_onset_s:
            return "validation"
        return "test"
    return "unused"


def bin_port(port: int) -> str:
    if port == 502:
        return "502"
    if port == -1:
        return "none"
    return "other"


def build_pair_token(row: pd.Series, token_schema: str) -> str:
    base = (
        "fc=" + str(int(row["function_code"]))
        + "|addr=" + str(int(row["address"]))
    )
    if token_schema == "fc_address":
        return base
    if token_schema == "fc_address_ip":
        return base + "|src=" + str(row["src_ip"])
    if token_schema == "fc_address_port":
        return base + "|dport=" + bin_port(int(row["dst_port"]))
    if token_schema == "fc_address_ip_port":
        return (
            base
            + "|src=" + str(row["src_ip"])
            + "|dport=" + bin_port(int(row["dst_port"]))
        )
    if token_schema == "fc_address_protocol_port":
        return (
            base
            + "|proto=" + str(row["protocol"])
            + "|dport=" + bin_port(int(row["dst_port"]))
        )
    if token_schema == "fc_address_protocol_port_ip":
        return (
            base
            + "|proto=" + str(row["protocol"])
            + "|dport=" + bin_port(int(row["dst_port"]))
            + "|src=" + str(row["src_ip"])
        )
    raise ValueError(f"Unsupported token_schema={token_schema}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="roll-script/rolling_labels.json",
        help="Path to the rolling-data labeling config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/rolling_fc_address",
        help="Directory where the processed rolling dataset will be written.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Window size in seconds.",
    )
    parser.add_argument(
        "--token-schema",
        default="fc_address",
        choices=[
            "fc_address",
            "fc_address_ip",
            "fc_address_port",
            "fc_address_ip_port",
            "fc_address_protocol_port",
            "fc_address_protocol_port_ip",
        ],
        help="How to compose the packet token used by downstream sequence models.",
    )
    parser.add_argument(
        "--session-attack-onset-s",
        type=float,
        default=None,
        help="For session_ics: offset in seconds where attacks begin. "
             "Packets before this * train_fraction \u2192 train, "
             "this * train_fraction to this \u2192 validation, "
             "after this \u2192 test.",
    )
    parser.add_argument(
        "--session-train-fraction",
        type=float,
        default=0.8,
        help="Fraction of the normal period used for training (rest = validation).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    by_pcap_counts: dict[str, dict[str, int]] = {}
    label_counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for pcap_cfg in config["pcaps"]:
        rows, counts = process_pcap(
            project_root,
            config["attacker_ip"],
            config["target_ip"],
            pcap_cfg,
        )
        all_rows.extend(rows)
        by_pcap_counts[str(pcap_cfg["name"])] = dict(counts)
        label_counts.update(counts)
        for row in rows:
            key = f"fc={row['function_code']},addr={row['address']}"
            feature_counts[row["label"]][key] += 1

    packet_dataset_path = output_dir / "rolling_packet_features_fc_address.csv"
    packet_df = pd.DataFrame(all_rows)
    onset = args.session_attack_onset_s
    frac = args.session_train_fraction
    packet_df["split"] = packet_df.apply(
        lambda row: assign_packet_split(row, onset, frac),
        axis=1,
    )
    packet_df["pair_token"] = packet_df.apply(build_pair_token, axis=1, token_schema=args.token_schema)
    write_csv(packet_dataset_path, packet_df.to_dict(orient="records"))
    packet_df["window_index"] = (
        packet_df["time_offset_s"] // args.window_seconds
    ).astype(int)

    grouped = (
        packet_df.groupby(["pcap_name", "window_index", "pair_token"])
        .size()
        .rename("count")
        .reset_index()
    )

    window_df = grouped.pivot_table(
        index=["pcap_name", "window_index"],
        columns="pair_token",
        values="count",
        fill_value=0,
    ).reset_index()

    label_df = (
        packet_df.groupby(["pcap_name", "window_index"], as_index=False)
        .agg(
            label=("label", lambda s: "anomaly" if (s != "normal").any() else "normal"),
            is_anomaly=("is_attack", "max"),
            attack_packet_count=("is_attack", "sum"),
            packet_count=("label", "size"),
            window_start_s=("time_offset_s", "min"),
            window_end_s=("time_offset_s", "max"),
        )
    )
    label_df["attack_packet_ratio"] = (
        label_df["attack_packet_count"] / label_df["packet_count"]
    )

    full_index_rows: list[dict[str, int | str]] = []
    for pcap_name, pcap_df in packet_df.groupby("pcap_name"):
        max_window = int((pcap_df["time_offset_s"].max() // args.window_seconds))
        for window_index in range(max_window + 1):
            full_index_rows.append(
                {
                    "pcap_name": str(pcap_name),
                    "window_index": int(window_index),
                }
            )

    full_index_df = pd.DataFrame(full_index_rows)
    window_df = full_index_df.merge(window_df, on=["pcap_name", "window_index"], how="left")
    window_df = full_index_df.merge(label_df, on=["pcap_name", "window_index"], how="left").merge(
        window_df,
        on=["pcap_name", "window_index"],
        how="left",
    )

    feature_columns = [column for column in window_df.columns if column.startswith("fc=")]
    for column in feature_columns:
        window_df[column] = window_df[column].fillna(0)

    window_df["label"] = window_df["label"].fillna("normal")
    window_df["is_anomaly"] = window_df["is_anomaly"].fillna(0).astype(int)
    window_df["attack_packet_count"] = window_df["attack_packet_count"].fillna(0).astype(int)
    window_df["packet_count"] = window_df["packet_count"].fillna(0).astype(int)
    window_df["attack_packet_ratio"] = window_df["attack_packet_ratio"].fillna(0.0)
    window_df["window_start_s"] = window_df["window_start_s"].fillna(window_df["window_index"] * args.window_seconds)
    window_df["window_end_s"] = window_df["window_end_s"].fillna(
        (window_df["window_index"] + 1) * args.window_seconds
    )

    window_df["split"] = window_df.apply(
        lambda row: assign_split_session(row, onset, frac),
        axis=1,
    )
    window_df = window_df.sort_values(["pcap_name", "window_index"]).reset_index(drop=True)

    window_dataset_path = output_dir / "rolling_window_features_fc_address.csv"
    window_df.to_csv(window_dataset_path, index=False)

    summary = {
        "target_ip": config["target_ip"],
        "attacker_ip": config["attacker_ip"],
        "window_seconds": args.window_seconds,
        "token_schema": args.token_schema,
        "packet_row_count": len(all_rows),
        "packet_label_counts": dict(label_counts),
        "per_pcap_packet_label_counts": by_pcap_counts,
        "top_feature_pairs_per_label": {
            label: counter.most_common(10) for label, counter in feature_counts.items()
        },
        "window_row_count": int(len(window_df)),
        "window_label_counts": window_df["label"].value_counts().to_dict(),
        "split_counts": window_df["split"].value_counts().to_dict(),
        "feature_columns": [column for column in window_df.columns if column.startswith("fc=")],
    }
    summary_path = output_dir / "rolling_preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote packet dataset to {packet_dataset_path}")
    print(f"Wrote window dataset to {window_dataset_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
