from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scapy.all import ICMP, IP, PcapReader, Raw, TCP  # type: ignore


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
    if ip_layer.dst != target_ip:
        return None

    protocol = "OTHER"
    src_port = -1
    dst_port = -1
    function_code = -1
    address = -1
    tcp_flags = ""
    payload_len = 0

    if TCP in packet:
        protocol = "TCP"
        tcp_layer = packet[TCP]
        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        tcp_flags = str(tcp_layer.flags)
        payload_len = len(bytes(tcp_layer.payload))
        if dst_port == 502 and Raw in packet:
            data = bytes(packet[Raw].load)
            if len(data) >= 10:
                function_code = int(data[7])
                address = int.from_bytes(data[8:10], byteorder="big")
    elif ICMP in packet:
        protocol = "ICMP"

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


def process_pcap(
    project_root: Path,
    target_ip: str,
    pcap_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    pcap_path = project_root / str(pcap_cfg["path"])
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    first_timestamp: float | None = None

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            features = extract_packet_features(packet, target_ip)
            if not features:
                continue

            packet_ts = float(packet.time)
            if first_timestamp is None:
                first_timestamp = packet_ts
            offset_s = packet_ts - first_timestamp
            label = label_for_offset(offset_s, pcap_cfg["segments"])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/packet_labels.json",
        help="Path to the packet labeling config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fc_address",
        help="Directory where the processed dataset will be written.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    output_dir = project_root / args.output_dir

    all_rows: list[dict[str, Any]] = []
    by_pcap_counts: dict[str, dict[str, int]] = {}
    label_counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for pcap_cfg in config["pcaps"]:
        rows, counts = process_pcap(project_root, config["target_ip"], pcap_cfg)
        all_rows.extend(rows)
        by_pcap_counts[str(pcap_cfg["name"])] = dict(counts)
        label_counts.update(counts)
        for row in rows:
            key = f"fc={row['function_code']},addr={row['address']}"
            feature_counts[row["label"]][key] += 1

    dataset_path = output_dir / "packet_features_fc_address.csv"
    write_csv(dataset_path, all_rows)

    summary = {
      "target_ip": config["target_ip"],
      "row_count": len(all_rows),
      "label_counts": dict(label_counts),
      "per_pcap_label_counts": by_pcap_counts,
      "top_feature_pairs_per_label": {
          label: counter.most_common(10) for label, counter in feature_counts.items()
      },
    }
    summary_path = output_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote dataset to {dataset_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
