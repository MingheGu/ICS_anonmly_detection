from __future__ import annotations

from lstm_window_common import build_parser, build_time_windows, prepare_run, write_outputs
from lstm_window_modes import run_sliding_retrain_mode


def main() -> None:
    parser = build_parser("Retrain packet-level LSTM for each sliding train/test window.")
    args = parser.parse_args()

    dataset_path, output_dir, df, token_to_idx, filter_summary = prepare_run(args)
    windows = build_time_windows(df, args.train_duration_s, args.test_duration_s, args.step_s)
    metrics_rows, scores_df, last_train_losses, last_val_losses = run_sliding_retrain_mode(
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
        execution_mode="sliding_retrain",
    )


if __name__ == "__main__":
    main()
