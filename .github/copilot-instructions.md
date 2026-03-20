# ICS Anomaly Detection Study - AI Agent Guidelines

## Project Overview
This is an industrial control systems (ICS) anomaly detection research project using machine learning on network packet data. The codebase analyzes Modbus protocol traffic to detect attacks like unauthorized writes and port scans.

**Architecture**: Two-stage pipeline - packet-level feature extraction (scapy) → window aggregation → anomaly detection models (IsolationForest for address features, LSTM autoencoder for sequences).

## Key Workflows

### Environment Setup
```bash
python3 -m venv .venv-ml
source .venv-ml/bin/activate
pip install -r requirements.txt
```

### FC Address Anomaly (Baseline)
```bash
python scripts/pre_process_fc_address.py
python scripts/pre_process_fc_address_windows.py
python scripts/train_fc_address_anomaly.py
```

### LSTM Sequence Anomaly
```bash
python scripts/pre_process_fc_address.py
python scripts/pre_process_fc_address_windows.py
python scripts/build_lstm_sequences.py
python scripts/train_lstm_anomaly.py
```

## Code Patterns

### Feature Extraction
- Use `scapy` for packet parsing, focus on IP/TCP/ICMP layers
- Target IP filtering: only process packets where `ip_layer.dst == target_ip` (from config)
- Modbus-specific: extract function codes and addresses from TCP port 502 payloads
- Window aggregation: group packets into time windows (see `pre_process_fc_address_windows.py`)

### Labeling Logic
- Load labels from `config/packet_labels.json` with time-based segments
- Refine coarse labels using packet features (attacker IP, ports, protocols)
- Normal traffic: `src_ip != attacker_ip or dst_ip != target_ip`
- Attack patterns: specific function codes, ports, TCP flags

### Model Training
- **FC Model**: `IsolationForest` on window-level aggregated features, trained only on normal data
- **LSTM Model**: Autoencoder with `StandardScaler` normalization, sequences of 10-20 windows
- Evaluation: separate train/validation/test splits, focus on attack PCAPs for testing
- Outputs: confusion matrices, ROC curves, precision/recall metrics

### File Organization
- `scripts/`: executable Python files with `argparse` CLIs
- `artifacts/{model}/`: model outputs, CSVs, trained models (.pt), JSON summaries
- `data/`: PCAP files organized by normal/attack
- `config/`: labeling configuration JSON

### Dependencies
- `scapy`: packet dissection
- `torch`: LSTM implementation
- `sklearn`: FC models, metrics, scaling
- `pandas/numpy`: data manipulation
- `matplotlib`: plotting

### Conventions
- Use `Path` from `pathlib` for file operations
- JSON for metadata/configuration, CSV for features, NPZ for sequences
- Type hints with `from __future__ import annotations`
- Argparse with sensible defaults pointing to `artifacts/` outputs
- Sequence labels: anomaly if any window in sequence is anomalous