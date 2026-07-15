"""Aggregate seed-level CEM results and execution diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloaded"
OUT = ROOT / "audit_artifacts" / "aggregate"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    result_files = sorted(DOWNLOADS.glob("**/experiment_results_clean.csv"))
    if not result_files:
        raise SystemExit("No experiment results found")
    frames = []
    for path in result_files:
        frame = pd.read_csv(path)
        if "base_seed" in frame:
            frame["seed"] = frame["base_seed"].astype(int)
        else:
            digits = "".join(ch for ch in str(path) if ch.isdigit())
            frame["seed"] = int(digits[-2:])
        frames.append(frame)
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(OUT / "all_seed_results.csv", index=False)

    metrics = [
        "test_return_pct", "test_excess_return_pct", "test_sharpe",
        "test_max_dd_pct", "test_win_rate_pct", "test_trades",
        "train_return_pct", "train_excess_return_pct", "train_sharpe",
        "train_max_dd_pct", "train_win_rate_pct", "train_trades",
    ]
    agg = all_results.groupby(["experiment", "benchmark"], as_index=False)[metrics].agg(
        ["mean", "std", "min", "max"]
    )
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
    wins = (all_results["test_excess_return_pct"] > 0).groupby(
        [all_results["experiment"], all_results["benchmark"]]
    ).agg(["sum", "count"]).reset_index()
    wins.columns = ["experiment", "benchmark", "test_excess_positive_seeds", "seed_count"]
    agg = agg.merge(wins, on=["experiment", "benchmark"], how="left")
    agg.to_csv(OUT / "seed_aggregate.csv", index=False)

    trade_files = sorted(DOWNLOADS.glob("**/experiment_trade_logs_clean/*.csv"))
    trade_frames = []
    for path in trade_files:
        frame = pd.read_csv(path)
        if "execution_model" in frame.columns:
            frame["source_file"] = str(path)
            trade_frames.append(frame)
    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        trades.to_csv(OUT / "detailed_execution_trades.csv", index=False)
        diag = trades.groupby(["benchmark", "experiment", "split"], as_index=False).agg(
            trades=("symbol", "size"),
            gap_fills=("stop_gap_fill", "sum"),
            missing_open_fallbacks=("stop_missing_open_fallback", "sum"),
            mean_pnl_pct=("pnl_pct", "mean"),
            net_pnl=("pnl", "sum"),
        )
        diag.to_csv(OUT / "execution_diagnostics.csv", index=False)
    else:
        diag = pd.DataFrame()

    summary = {
        "result_files": len(result_files),
        "rows": len(all_results),
        "seeds": sorted(all_results.seed.unique().tolist()),
        "experiments": sorted(all_results.experiment.unique().tolist()),
        "benchmarks": sorted(all_results.benchmark.unique().tolist()),
        "missing_open_fallbacks": int(diag["missing_open_fallbacks"].sum()) if not diag.empty else None,
        "gap_fills": int(diag["gap_fills"].sum()) if not diag.empty else None,
    }
    (OUT / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
