# ICS Anomaly Study

Independent research workspace extracted from `labshock`.

## Layout

- `scripts/`: preprocessing and model code
- `config/`: labeling configuration
- `data/`: copied pcap files used for experiments
- `artifacts/`: generated datasets, plots, metrics, and trained models

## Setup

From this directory:

```bash
python3 -m venv .venv-ml
source .venv-ml/bin/activate
pip install -r requirements.txt
```

## Run Baseline

```bash
python scripts/pre_process_fc_address.py
python scripts/pre_process_fc_address_windows.py
python scripts/train_fc_address_anomaly.py
```

## Run LSTM

```bash
python scripts/pre_process_fc_address.py
python scripts/pre_process_fc_address_windows.py
python scripts/build_lstm_sequences.py
python scripts/train_lstm_anomaly.py
```
