from __future__ import annotations

import sys

from train_lstm_fixed import main as fixed_main
from train_lstm_sliding_retrain import main as sliding_retrain_main


def _extract_fixed_train_end(argv: list[str]) -> float:
    for index, arg in enumerate(argv):
        if arg.startswith("--fixed-train-end-s="):
            return float(arg.split("=", 1)[1])
        if arg == "--fixed-train-end-s" and index + 1 < len(argv):
            return float(argv[index + 1])
    return 0.0


def main() -> None:
    fixed_train_end_s = _extract_fixed_train_end(sys.argv[1:])
    if fixed_train_end_s > 0:
        fixed_main()
    else:
        sliding_retrain_main()


if __name__ == "__main__":
    main()
