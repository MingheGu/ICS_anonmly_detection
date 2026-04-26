from __future__ import annotations

from lstm_window_common import build_parser, prepare_run, write_outputs
from lstm_window_modes import run_static_mode


def main() -> None:
    parser = build_parser("Train once and evaluate a single static test window with packet-level LSTM.")
    args = parser.parse_args()

    dataset_path, output_dir, df, token_to_idx, filter_summary = prepare_run(args)
    windows, metrics_rows, scores_df, last_train_losses, last_val_losses = run_static_mode(df, token_to_idx, args)
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
        execution_mode="static_single_window",
    )


if __name__ == "__main__":
    main()
