from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scapy.all import ICMP, IP, PcapReader, Raw, TCP  # type: ignore


def load_defaults(config_path: Path) -> tuple[str, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return str(config["attacker_ip"]), str(config["target_ip"])


def detect_segments(
    pcap_path: Path,
    attacker_ip: str,
    target_ip: str,
    project_root: Path,
) -> dict[str, Any]:
    first_timestamp: float | None = None
    scan_offsets: list[float] = []
    write_offsets: list[float] = []
    write_function_codes: dict[int, int] = {}
    packet_count = 0

    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP not in packet:
                continue

            packet_count += 1
            packet_ts = float(packet.time)
            if first_timestamp is None:
                first_timestamp = packet_ts
            offset_s = packet_ts - first_timestamp

            ip_layer = packet[IP]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            is_attacker_flow = src_ip == attacker_ip and dst_ip == target_ip

            if ICMP in packet and is_attacker_flow:
                scan_offsets.append(offset_s)

            if TCP not in packet:
                continue

            tcp_layer = packet[TCP]
            dst_port = int(tcp_layer.dport)
            tcp_flags = str(tcp_layer.flags)
            payload_len = len(bytes(tcp_layer.payload))

            if is_attacker_flow and dst_port != 502:
                is_syn_probe = "S" in tcp_flags and "A" not in tcp_flags
                is_non_modbus_payload = payload_len > 0
                if is_syn_probe or is_non_modbus_payload:
                    scan_offsets.append(offset_s)

            if not (is_attacker_flow and dst_port == 502 and Raw in packet):
                continue

            data = bytes(packet[Raw].load)
            if len(data) < 10:
                continue

            function_code = int(data[7])
            if function_code in {5, 6}:
                write_offsets.append(offset_s)
                write_function_codes[function_code] = write_function_codes.get(function_code, 0) + 1

    if first_timestamp is None:
        raise ValueError(f"No IP packets found in {pcap_path}")

    def summarize(offsets: list[float]) -> dict[str, float | int | None]:
        if not offsets:
            return {"count": 0, "start_offset_s": None, "end_offset_s": None}
        return {
            "count": len(offsets),
            "start_offset_s": round(min(offsets), 6),
            "end_offset_s": round(max(offsets), 6),
        }

    scan_summary = summarize(scan_offsets)
    write_summary = summarize(write_offsets)
    return {
        "pcap_path": str(pcap_path),
        "packet_count": packet_count,
        "attacker_ip": attacker_ip,
        "target_ip": target_ip,
        "scan": scan_summary,
        "write": write_summary,
        "write_function_code_counts": write_function_codes,
        "rolling_labels_entry": {
            "name": pcap_path.stem,
            "path": str(pcap_path.relative_to(project_root)),
            "segments": [
                {
                    "start": 0.0,
                    "end": scan_summary["start_offset_s"],
                    "label": "normal",
                },
                {
                    "start": scan_summary["start_offset_s"],
                    "end": scan_summary["end_offset_s"],
                    "label": "attack_scan",
                },
                {
                    "start": scan_summary["end_offset_s"],
                    "end": write_summary["start_offset_s"],
                    "label": "normal",
                },
                {
                    "start": write_summary["start_offset_s"],
                    "end": write_summary["end_offset_s"],
                    "label": "attack_write",
                },
                {
                    "start": write_summary["end_offset_s"],
                    "end": None,
                    "label": "normal",
                },
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pcap",
        default="rolling-data/continue_data/mixed_long_conti.pcap",
        help="PCAP path relative to the project root.",
    )
    parser.add_argument(
        "--config",
        default="roll-script/rolling_labels.json",
        help="Config used to load default attacker and target IPs.",
    )
    parser.add_argument("--attacker-ip", default=None)
    parser.add_argument("--target-ip", default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    attacker_ip, target_ip = load_defaults(project_root / args.config)
    attacker_ip = args.attacker_ip or attacker_ip
    target_ip = args.target_ip or target_ip

    result = detect_segments(project_root / args.pcap, attacker_ip, target_ip, project_root)
    result_json = json.dumps(result, indent=2)

    if args.output_json:
        output_path = project_root / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json, encoding="utf-8")

    print(result_json)


if __name__ == "__main__":
    main()
