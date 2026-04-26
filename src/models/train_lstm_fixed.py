from __future__ import annotations

from lstm_window_common import (
    build_fixed_train_windows,
    build_parser,
    prepare_run,
    write_outputs,
)
from lstm_window_modes import run_fixed_train_mode


def main() -> None:
    parser = build_parser("Train once on a fixed prefix and score multiple sliding test windows with packet-level LSTM.")
    args = parser.parse_args()
    if args.fixed_train_end_s <= 0:
        raise ValueError("--fixed-train-end-s must be > 0 for train-once sliding-test mode.")

    dataset_path, output_dir, df, token_to_idx, filter_summary = prepare_run(args)
    windows = build_fixed_train_windows(df, args.fixed_train_end_s, args.test_duration_s, args.step_s)
    metrics_rows, scores_df, last_train_losses, last_val_losses = run_fixed_train_mode(
        df, windows, token_to_idx, args
    )
    write_outputs(
        dataset_path=dataset_path,
        output_dir=output_dir,
        df=df,
        windows=windows,
        metrics_rows=metrics_rows,
        scores_df=scores_df,
        last_train_losses=last_train_losses,
        last_val_losses=last_val_losses,
        token_to_idx=token_to_idx,
        filter_summary=filter_summary,
        args=args,
        execution_mode="fixed_train",
    )


if __name__ == "__main__":
    main()
