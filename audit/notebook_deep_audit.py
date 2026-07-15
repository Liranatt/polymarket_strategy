from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def num(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                pass
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def load_result_rows() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in sorted((ROOT / "runs").glob("*/experiment_results_clean.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "run_id", path.parent.name)
        records.append(frame)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True, sort=False)


def metric_table(results: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "run_id", "benchmark", "base_seed", "policy_scope", "train_fit_candidates",
        "train_total_return", "train_benchmark_return", "train_excess_return",
        "train_sharpe", "train_max_dd", "train_n_trades", "train_win_rate",
        "test_total_return", "test_benchmark_return", "test_excess_return",
        "test_sharpe", "test_max_dd", "test_n_trades", "test_win_rate",
        "policy_json", "policy_snapshot_json",
    ]
    return results[[c for c in wanted if c in results.columns]].copy()


def next_bar_close(bars: list[tuple], day: pd.Timestamp, strict: bool) -> tuple[pd.Timestamp, float] | None:
    d = pd.Timestamp(day)
    if d.tz is None:
        d = d.tz_localize("UTC")
    else:
        d = d.tz_convert("UTC")
    d = d.normalize()
    for ts, _high, _low, close in bars:
        t = pd.Timestamp(ts)
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        if (t.normalize() > d) if strict else (t.normalize() >= d):
            return t.normalize(), float(close)
    return None


def cluster_bootstrap(trades: pd.DataFrame, rng: np.random.Generator, reps: int = 5000) -> dict[str, float]:
    work = trades.copy()
    work["entry_day"] = pd.to_datetime(work["entry_date"], utc=True).dt.normalize()
    grouped = work.groupby("entry_day", sort=True)["pnl"].sum()
    vals = grouped.to_numpy(dtype=float)
    if len(vals) == 0:
        return {}
    samples = rng.choice(vals, size=(reps, len(vals)), replace=True).sum(axis=1)
    return {
        "clusters": float(len(vals)),
        "observed_total": float(vals.sum()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "prob_positive": float(np.mean(samples > 0)),
    }


def analyse_trades(benchmark: str, prices: dict[str, list[tuple]]) -> dict[str, Any]:
    path = ROOT / "notebooks" / "outputs" / f"{benchmark.lower()}_trades.csv"
    trades = pd.read_csv(path)
    for col in ("pnl", "gross_pnl", "txn_cost", "pnl_pct", "_asset_entry_notional"):
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], utc=True)
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], utc=True)

    recon = (trades["gross_pnl"] - trades["txn_cost"] - trades["pnl"]).abs().max()
    wins = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] < 0).sum())
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf

    by_day = trades.groupby(trades["entry_date"].dt.normalize())["pnl"].sum().sort_values(ascending=False)
    total = float(trades["pnl"].sum())
    top5_day_share = float(by_day.head(5).sum() / total) if total else math.nan
    abs_day = by_day.abs().to_numpy(dtype=float)
    effective_days = float(abs_day.sum() ** 2 / np.square(abs_day).sum()) if np.square(abs_day).sum() else 0.0

    by_month = trades.groupby(trades["entry_date"].dt.to_period("M"))["pnl"].sum()
    by_month.index = by_month.index.astype(str)

    delayed_returns: list[float] = []
    delayed_both_returns: list[float] = []
    canonical_gross_returns: list[float] = []
    impossible = 0
    for row in trades.itertuples(index=False):
        bars = prices.get(str(row.symbol), [])
        entry_next = next_bar_close(bars, row.entry_date, strict=True)
        exit_same = next_bar_close(bars, row.exit_date, strict=False)
        exit_next = next_bar_close(bars, row.exit_date, strict=True)
        if not entry_next or not exit_same:
            impossible += 1
            continue
        if entry_next[0] > exit_same[0]:
            impossible += 1
            continue
        canonical_gross_returns.append(float(row.exit_price) / float(row.entry_price) - 1.0)
        delayed_returns.append(exit_same[1] / entry_next[1] - 1.0)
        if exit_next and exit_next[0] > entry_next[0]:
            delayed_both_returns.append(exit_next[1] / entry_next[1] - 1.0)

    rng = np.random.default_rng(20260715 + (0 if benchmark == "SPY" else 1))
    boot = cluster_bootstrap(trades, rng)

    result = {
        "benchmark": benchmark,
        "n": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": 100.0 * wins / len(trades),
        "net_pnl": total,
        "mean_pnl_pct": float(trades["pnl_pct"].mean()),
        "median_pnl_pct": float(trades["pnl_pct"].median()),
        "profit_factor": pf,
        "max_reconciliation_error": float(recon),
        "entry_days": int(trades["entry_date"].dt.normalize().nunique()),
        "effective_entry_days_abs_pnl": effective_days,
        "top5_entry_days_pnl_share": top5_day_share,
        "monthly_pnl": {str(k): float(v) for k, v in by_month.items()},
        "cluster_bootstrap": boot,
        "timing_stress": {
            "usable_trades": int(len(delayed_returns)),
            "unusable": impossible,
            "canonical_mean_gross_return_pct": 100.0 * float(np.mean(canonical_gross_returns)) if canonical_gross_returns else math.nan,
            "next_bar_entry_same_exit_mean_gross_return_pct": 100.0 * float(np.mean(delayed_returns)) if delayed_returns else math.nan,
            "next_bar_entry_and_exit_mean_gross_return_pct": 100.0 * float(np.mean(delayed_both_returns)) if delayed_both_returns else math.nan,
        },
    }
    return result


def main() -> None:
    report: list[str] = []
    report.append("NOTEBOOK-SPECIFIC DEEP AUDIT")
    report.append("=" * 80)

    executed_path = OUT / "executed_standard_cem_strategy.ipynb"
    notebook = nbformat.read(executed_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = []
    stdout_parts: list[str] = []
    source_parts: list[str] = []
    for idx, cell in enumerate(code_cells):
        source_parts.append(f"\n# CODE CELL {idx}\n{cell.source}\n")
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(output)
            text = output.get("text")
            if text:
                stdout_parts.append(str(text))
    source = "\n".join(source_parts)
    stdout = "\n".join(stdout_parts)
    (OUT / "notebook_code_cells.py").write_text(source, encoding="utf-8")
    (OUT / "notebook_stdout.txt").write_text(stdout, encoding="utf-8")
    report.append(f"Executed code cells: {len(code_cells)}; notebook errors: {len(errors)}")
    report.append(f"Execution counts complete: {all(cell.get('execution_count') is not None for cell in code_cells)}")

    interesting = []
    for line in source.splitlines():
        if re.search(r"optimize_cem|subprocess|read_parquet|read_pickle|read_csv|OOS|oos|split|test|train", line):
            interesting.append(line.strip())
    (OUT / "notebook_relevant_lines.txt").write_text("\n".join(interesting), encoding="utf-8")

    input_paths = [
        ROOT / "data" / "candidates_audit_clean.parquet",
        ROOT / "data" / "prices.pkl",
        ROOT / "data" / "probs.pkl",
        ROOT / "data" / "polarity_labels.json",
    ]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    report.append("Input artifact SHA-256 values recorded; notebook did not need network access.")

    candidates = pd.read_parquet(ROOT / "data" / "candidates_audit_clean.parquet")
    candidates["t_theta"] = pd.to_datetime(candidates["t_theta"], utc=True)
    candidates["t_e"] = pd.to_datetime(candidates["t_e"], utc=True)
    split_counts = candidates["split"].astype(str).value_counts().to_dict()
    report.append(f"Candidates: {len(candidates)}; split counts: {split_counts}")
    report.append(
        f"Candidate dates: {candidates['t_theta'].min()} to {candidates['t_theta'].max()}; "
        f"invalid t_e<t_theta: {int((candidates['t_e'] < candidates['t_theta']).sum())}"
    )
    key_dupes = int(candidates.duplicated(["market_id", "symbol"]).sum())
    report.append(f"Duplicate market_id x symbol rows: {key_dupes}")

    train = candidates[candidates["split"].astype(str).str.lower() == "train"]
    test = candidates[candidates["split"].astype(str).str.lower().isin(["test", "val"])]
    market_overlap = len(set(train["market_id"].astype(str)) & set(test["market_id"].astype(str)))
    question_overlap = len(set(train.get("question", pd.Series(dtype=str)).astype(str)) & set(test.get("question", pd.Series(dtype=str)).astype(str)))
    symbol_overlap = len(set(train["symbol"].astype(str)) & set(test["symbol"].astype(str)))
    report.append(f"Train/Test overlap: market_id={market_overlap}, exact question={question_overlap}, symbols={symbol_overlap}")

    suspicious = [
        c for c in candidates.columns
        if re.search(r"future|forward|target|label|outcome|return|pnl|exit|resolved|resolution_price", c, re.I)
    ]
    report.append(f"Leakage-sensitive column names present in candidate artifact: {suspicious}")

    with (ROOT / "data" / "prices.pkl").open("rb") as handle:
        prices = pickle.load(handle)
    with (ROOT / "data" / "probs.pkl").open("rb") as handle:
        probs = pickle.load(handle)

    price_times = Counter()
    for bars in prices.values():
        for ts, *_ in bars:
            price_times[str(pd.Timestamp(ts).time())] += 1
    prob_times = Counter()
    for points in probs.values():
        for ts, _ in points:
            prob_times[str(pd.Timestamp(ts).time())] += 1
    theta_times = Counter(str(ts.time()) for ts in candidates["t_theta"])
    report.append(f"Most common price timestamp times: {price_times.most_common(5)}")
    report.append(f"Most common probability timestamp times: {prob_times.most_common(5)}")
    report.append(f"Most common t_theta times: {theta_times.most_common(5)}")

    results = load_result_rows()
    metrics = metric_table(results)
    metrics.to_csv(OUT / "all_run_metrics.csv", index=False)
    report.append(f"Fresh result rows found: {len(results)} across runs {sorted(results['run_id'].unique()) if not results.empty else []}")

    seed_rows = results[results["run_id"].str.startswith("notebook_seed_")].copy() if not results.empty else pd.DataFrame()
    seed_summary: dict[str, Any] = {}
    if not seed_rows.empty:
        for benchmark, group in seed_rows.groupby("benchmark"):
            seed_summary[str(benchmark)] = {
                "seeds": sorted(group["base_seed"].astype(int).tolist()),
                "test_return_mean": float(group["test_total_return"].mean()),
                "test_return_min": float(group["test_total_return"].min()),
                "test_return_max": float(group["test_total_return"].max()),
                "test_sharpe_mean": float(group["test_sharpe"].mean()),
                "test_sharpe_min": float(group["test_sharpe"].min()),
                "test_sharpe_max": float(group["test_sharpe"].max()),
            }
            report.append(f"Seed sensitivity {benchmark}: {seed_summary[str(benchmark)]}")

    variant_summary: dict[str, Any] = {}
    for run_id in ("notebook_delay1d", "notebook_circular_placebo"):
        group = results[results["run_id"] == run_id] if not results.empty else pd.DataFrame()
        if group.empty:
            continue
        variant_summary[run_id] = {}
        for _, row in group.iterrows():
            benchmark = str(row["benchmark"])
            variant_summary[run_id][benchmark] = {
                "test_return": num(row, "test_total_return"),
                "test_excess": num(row, "test_excess_return"),
                "test_sharpe": num(row, "test_sharpe"),
                "test_max_dd": num(row, "test_max_dd"),
                "test_win_rate": num(row, "test_win_rate"),
                "test_n_trades": num(row, "test_n_trades"),
            }
        report.append(f"Variant {run_id}: {variant_summary[run_id]}")

    trade_analysis = {benchmark: analyse_trades(benchmark, prices) for benchmark in ("SPY", "QQQ")}
    for benchmark, stats in trade_analysis.items():
        report.append("-" * 80)
        report.append(
            f"{benchmark} TEST trades: n={stats['n']}, wins={stats['wins']}, "
            f"win_rate={stats['win_rate_pct']:.4f}%, net_pnl=${stats['net_pnl']:.2f}, "
            f"PF={stats['profit_factor']:.4f}"
        )
        report.append(
            f"Concentration: entry_days={stats['entry_days']}, effective_days={stats['effective_entry_days_abs_pnl']:.2f}, "
            f"top-5 day PnL share={stats['top5_entry_days_pnl_share']:.4f}"
        )
        report.append(f"Monthly PnL: {stats['monthly_pnl']}")
        report.append(f"Entry-day cluster bootstrap: {stats['cluster_bootstrap']}")
        report.append(f"Trade-level next-bar timing stress: {stats['timing_stress']}")
        report.append(f"Max PnL reconciliation error: {stats['max_reconciliation_error']}")

    fitted_path = ROOT / "notebooks" / "outputs" / "cem_fitted_parameters.json"
    fitted = json.loads(fitted_path.read_text(encoding="utf-8")) if fitted_path.exists() else {}
    report.append(f"Fitted parameters: {fitted}")

    payload = {
        "notebook": {
            "code_cells": len(code_cells),
            "errors": len(errors),
            "all_execution_counts_present": all(cell.get("execution_count") is not None for cell in code_cells),
        },
        "input_hashes": hashes,
        "candidate": {
            "rows": int(len(candidates)),
            "split_counts": {str(k): int(v) for k, v in split_counts.items()},
            "duplicate_market_symbol": key_dupes,
            "invalid_time_order": int((candidates["t_e"] < candidates["t_theta"]).sum()),
            "market_overlap": market_overlap,
            "question_overlap": question_overlap,
            "symbol_overlap": symbol_overlap,
            "suspicious_columns": suspicious,
        },
        "timestamp_times": {
            "price": price_times.most_common(20),
            "probability": prob_times.most_common(20),
            "t_theta": theta_times.most_common(20),
        },
        "seed_summary": seed_summary,
        "variant_summary": variant_summary,
        "trade_analysis": trade_analysis,
        "fitted_parameters": fitted,
    }
    (OUT / "notebook_deep_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (OUT / "notebook_deep_audit.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

    if errors:
        raise SystemExit("Executed notebook contains error outputs")
    if any(stats["max_reconciliation_error"] > 0.02 for stats in trade_analysis.values()):
        raise SystemExit("PnL reconciliation failure")


if __name__ == "__main__":
    main()
