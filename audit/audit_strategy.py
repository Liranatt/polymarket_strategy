from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "audit_ci"


def finite(x: Any) -> float | None:
    try:
        y = float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None


def pct(x: Any, digits: int = 4) -> float | None:
    y = finite(x)
    return None if y is None else round(100.0 * y, digits)


def sharpe_from_equity(s: pd.Series) -> float | None:
    r = pd.to_numeric(s, errors="coerce").pct_change().dropna()
    if len(r) < 2 or float(r.std(ddof=1)) <= 0:
        return None
    return float(np.sqrt(252.0) * r.mean() / r.std(ddof=1))


def max_drawdown_pct(s: pd.Series) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return None
    dd = x / x.cummax() - 1.0
    return float(dd.min() * 100.0)


def safe_group(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    if key not in df.columns or "pnl" not in df.columns:
        return []
    z = df.copy()
    z[key] = z[key].fillna("<missing>").astype(str)
    out = (
        z.groupby(key, dropna=False)
        .agg(n=("pnl", "size"), pnl=("pnl", "sum"), mean_pnl=("pnl", "mean"), win_rate=("pnl", lambda x: float((x > 0).mean())))
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    out["pnl"] = out["pnl"].round(2)
    out["mean_pnl"] = out["mean_pnl"].round(2)
    out["win_rate_pct"] = (100.0 * out.pop("win_rate")).round(2)
    return out.head(20).to_dict("records")


def read_fresh_trade_files() -> list[Path]:
    d = RUN / "experiment_trade_logs_clean"
    return sorted(p for p in d.glob("*.csv") if "test" in p.stem.lower())


def read_fresh_equity_files() -> list[Path]:
    d = RUN / "experiment_equity_logs_clean"
    return sorted(p for p in d.glob("*.csv") if "test" in p.stem.lower())


def benchmark_from_path(path: Path) -> str:
    s = path.stem.upper()
    if "SPY" in s:
        return "SPY"
    if "QQQ" in s:
        return "QQQ"
    return path.stem


def audit_candidates() -> dict[str, Any]:
    path = ROOT / "data" / "candidates_audit_clean.parquet"
    df = pd.read_parquet(path)
    for c in ("t_theta", "t_e"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    key_dups = int(df.duplicated(["market_id", "symbol"], keep=False).sum()) if {"market_id", "symbol"}.issubset(df.columns) else None
    invalid_order = int((df["t_e"] < df["t_theta"]).fillna(False).sum())
    missing_required = {c: int(df[c].isna().sum()) for c in ["market_id", "symbol", "t_theta", "t_e", "split", "feat_connection_strength"] if c in df.columns}
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_market_symbol_rows": key_dups,
        "invalid_t_e_before_t_theta": invalid_order,
        "missing_required": missing_required,
        "split_counts": {str(k): int(v) for k, v in df["split"].astype(str).value_counts(dropna=False).to_dict().items()},
        "t_theta_min": str(df["t_theta"].min()),
        "t_theta_max": str(df["t_theta"].max()),
        "t_e_min": str(df["t_e"].min()),
        "t_e_max": str(df["t_e"].max()),
    }
    if "cem_eligible" in df.columns:
        result["cem_eligible_counts"] = {str(k): int(v) for k, v in df["cem_eligible"].value_counts(dropna=False).to_dict().items()}
    return result


def audit_binary_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("prices.pkl", "probs.pkl"):
        path = ROOT / "data" / name
        with path.open("rb") as f:
            obj = pickle.load(f)
        lens = [len(v) for v in obj.values()]
        out[name] = {
            "type": type(obj).__name__,
            "keys": int(len(obj)),
            "empty_series": int(sum(n == 0 for n in lens)),
            "min_points": int(min(lens)) if lens else 0,
            "median_points": float(np.median(lens)) if lens else 0.0,
            "max_points": int(max(lens)) if lens else 0,
        }
    return out


def audit_results() -> dict[str, Any]:
    result_path = RUN / "experiment_results_clean.csv"
    results = pd.read_csv(result_path)
    return {
        "rows": int(len(results)),
        "columns": list(results.columns),
        "records": results.to_dict("records"),
    }


def audit_trades(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    for c in ("entry_date", "exit_date", "candidate_t_theta", "candidate_t_e", "_entry_ts", "_exit_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in ("pnl", "gross_pnl", "txn_cost", "pnl_pct", "return_pct", "entry_prob", "entry_price", "exit_price", "_qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pnl_identity_max_abs = None
    if {"pnl", "gross_pnl", "txn_cost"}.issubset(df.columns):
        pnl_identity_max_abs = float((df["pnl"] - (df["gross_pnl"] - df["txn_cost"])).abs().max())

    chronological_bad = int((df["exit_date"] < df["entry_date"]).fillna(False).sum()) if {"entry_date", "exit_date"}.issubset(df.columns) else None
    after_endpoint = int((df["exit_date"] >= df["candidate_t_e"]).fillna(False).sum()) if {"exit_date", "candidate_t_e"}.issubset(df.columns) else None
    before_signal = int((df["entry_date"] < df["candidate_t_theta"]).fillna(False).sum()) if {"entry_date", "candidate_t_theta"}.issubset(df.columns) else None

    total_pnl = float(df["pnl"].sum()) if "pnl" in df else 0.0
    sorted_pnl = df["pnl"].sort_values(ascending=False) if "pnl" in df else pd.Series(dtype=float)
    top5_share = float(sorted_pnl.head(5).sum() / total_pnl) if total_pnl != 0 else None
    top10_share = float(sorted_pnl.head(10).sum() / total_pnl) if total_pnl != 0 else None
    wins = df.loc[df["pnl"] > 0, "pnl"] if "pnl" in df else pd.Series(dtype=float)
    losses = df.loc[df["pnl"] < 0, "pnl"] if "pnl" in df else pd.Series(dtype=float)

    if {"entry_date", "exit_date"}.issubset(df.columns):
        holding = (df["exit_date"] - df["entry_date"]).dt.total_seconds() / 86400.0
    else:
        holding = pd.Series(dtype=float)

    if "entry_date" in df.columns:
        df["entry_month"] = df["entry_date"].dt.strftime("%Y-%m")

    return {
        "file": str(path.relative_to(ROOT)),
        "benchmark": benchmark_from_path(path),
        "n_trades": int(len(df)),
        "win_rate_pct": round(float((df["pnl"] > 0).mean() * 100.0), 4) if len(df) and "pnl" in df else None,
        "total_net_pnl": round(total_pnl, 2),
        "gross_profit": round(float(wins.sum()), 2),
        "gross_loss": round(float(losses.sum()), 2),
        "profit_factor": round(float(wins.sum() / abs(losses.sum())), 4) if float(losses.sum()) < 0 else None,
        "mean_pnl": round(float(df["pnl"].mean()), 2) if len(df) and "pnl" in df else None,
        "median_pnl": round(float(df["pnl"].median()), 2) if len(df) and "pnl" in df else None,
        "total_txn_cost": round(float(df["txn_cost"].sum()), 2) if "txn_cost" in df else None,
        "txn_cost_as_pct_gross_profit": round(float(df["txn_cost"].sum() / wins.sum() * 100.0), 4) if "txn_cost" in df and float(wins.sum()) > 0 else None,
        "pnl_identity_max_abs_error": pnl_identity_max_abs,
        "entries_before_signal": before_signal,
        "exits_before_entries": chronological_bad,
        "exits_on_or_after_endpoint": after_endpoint,
        "duplicate_market_symbol_trade_rows": int(df.duplicated(["market_id", "symbol"], keep=False).sum()) if {"market_id", "symbol"}.issubset(df.columns) else None,
        "median_holding_calendar_days": round(float(holding.median()), 4) if not holding.empty else None,
        "top_5_trade_share_of_net_pnl_pct": pct(top5_share),
        "top_10_trade_share_of_net_pnl_pct": pct(top10_share),
        "by_month": safe_group(df, "entry_month"),
        "by_archetype": safe_group(df, "archetype"),
        "by_exit_reason": safe_group(df, "realized_exit_reason" if "realized_exit_reason" in df.columns else "exit_reason"),
        "top_symbols": safe_group(df, "symbol"),
        "top_trades": df.nlargest(10, "pnl")[[c for c in ["market_id", "symbol", "question", "entry_date", "exit_date", "pnl", "pnl_pct", "txn_cost", "archetype", "realized_exit_reason"] if c in df.columns]].astype({"entry_date": "string", "exit_date": "string"}, errors="ignore").to_dict("records") if "pnl" in df else [],
        "worst_trades": df.nsmallest(10, "pnl")[[c for c in ["market_id", "symbol", "question", "entry_date", "exit_date", "pnl", "pnl_pct", "txn_cost", "archetype", "realized_exit_reason"] if c in df.columns]].astype({"entry_date": "string", "exit_date": "string"}, errors="ignore").to_dict("records") if "pnl" in df else [],
    }


def audit_equity(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    equity = pd.to_numeric(df["equity"], errors="coerce")
    bench = pd.to_numeric(df["benchmark_equity"], errors="coerce")
    sr = equity.pct_change().dropna()
    br = bench.pct_change().dropna()
    aligned = pd.concat([sr.rename("s"), br.rename("b")], axis=1).dropna()
    beta = None
    corr = None
    annualized_alpha = None
    if len(aligned) > 2 and float(aligned["b"].var(ddof=1)) > 0:
        beta = float(aligned.cov().loc["s", "b"] / aligned["b"].var(ddof=1))
        corr = float(aligned.corr().loc["s", "b"])
        annualized_alpha = float((aligned["s"].mean() - beta * aligned["b"].mean()) * 252.0 * 100.0)
    return {
        "file": str(path.relative_to(ROOT)),
        "benchmark": benchmark_from_path(path),
        "days": int(len(df)),
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
        "portfolio_return_pct": round(float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0), 4),
        "benchmark_return_pct": round(float((bench.iloc[-1] / bench.iloc[0] - 1.0) * 100.0), 4),
        "excess_return_pct": round(float((equity.iloc[-1] / equity.iloc[0] - bench.iloc[-1] / bench.iloc[0]) * 100.0), 4),
        "sharpe": round(sharpe_from_equity(equity), 4) if sharpe_from_equity(equity) is not None else None,
        "benchmark_sharpe": round(sharpe_from_equity(bench), 4) if sharpe_from_equity(bench) is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct(equity), 4) if max_drawdown_pct(equity) is not None else None,
        "benchmark_max_drawdown_pct": round(max_drawdown_pct(bench), 4) if max_drawdown_pct(bench) is not None else None,
        "daily_return_correlation": round(corr, 4) if corr is not None else None,
        "daily_beta": round(beta, 4) if beta is not None else None,
        "annualized_daily_alpha_pct": round(annualized_alpha, 4) if annualized_alpha is not None else None,
        "max_open_positions": int(pd.to_numeric(df.get("open_positions", pd.Series([0])), errors="coerce").max()),
        "min_cash": round(float(pd.to_numeric(df.get("cash", pd.Series([0.0])), errors="coerce").min()), 4),
    }


def main() -> None:
    report: dict[str, Any] = {
        "candidate_artifact": audit_candidates(),
        "binary_artifacts": audit_binary_artifacts(),
        "fresh_results": audit_results(),
        "trade_audits": [audit_trades(p) for p in read_fresh_trade_files()],
        "equity_audits": [audit_equity(p) for p in read_fresh_equity_files()],
    }

    failures: list[str] = []
    if report["candidate_artifact"]["invalid_t_e_before_t_theta"] != 0:
        failures.append("candidate rows with t_e < t_theta")
    for t in report["trade_audits"]:
        if t["entries_before_signal"] not in (0, None):
            failures.append(f"{t['benchmark']}: entries before signal")
        if t["exits_before_entries"] not in (0, None):
            failures.append(f"{t['benchmark']}: exits before entries")
        if t["exits_on_or_after_endpoint"] not in (0, None):
            failures.append(f"{t['benchmark']}: exits on/after endpoint")
        if t["pnl_identity_max_abs_error"] is not None and t["pnl_identity_max_abs_error"] > 0.02:
            failures.append(f"{t['benchmark']}: pnl identity mismatch")
    report["hard_invariant_failures"] = failures

    out = ROOT / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("AUDIT_JSON_START")
    print(json.dumps(report, indent=2, default=str))
    print("AUDIT_JSON_END")
    if failures:
        raise SystemExit("Hard invariant failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
