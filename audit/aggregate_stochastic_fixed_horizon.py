from __future__ import annotations

import argparse
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from audit.stochastic_fixed_horizon import (
    CANDIDATES, PRICES, PROBS, MONTHS, TEST_END, TRAIN_END, THETA_GRID,
    Policy, TradeCache, asdict, make_price_frame, select_policy, evaluate_policy,
    commission, SLIPPAGE, SEC_FEE, START_CAPITAL, PORTFOLIO_SLOTS,
)


def mode_int(s: pd.Series) -> int:
    counts = Counter(int(x) for x in s)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def consensus_policy(g: pd.DataFrame) -> Policy:
    weights = np.array([g.w_prob.median(), g.w_connection.median(), g.w_slope.median(), g.w_runup.median()], dtype=float)
    weights = np.maximum(weights, 1e-6); weights /= weights.sum()
    theta_raw = float(g.theta.median())
    theta = float(THETA_GRID[np.abs(THETA_GRID - theta_raw).argmin()])
    min_d = float(g.min_days_to_resolution.median())
    max_d = max(float(g.max_days_to_resolution.median()), min_d + 3.0)
    return Policy(
        theta=theta, confirm_points=mode_int(g.confirm_points),
        max_prob_surge=float(g.max_prob_surge.median()),
        max_price_runup=float(g.max_price_runup.median()),
        min_connection=float(g.min_connection.median()),
        min_days_to_resolution=min_d, max_days_to_resolution=max_d,
        max_event_assets=mode_int(g.max_event_assets), max_day_entries=mode_int(g.max_day_entries),
        w_prob=float(weights[0]), w_connection=float(weights[1]),
        w_slope=float(weights[2]), w_runup=float(weights[3]),
    )


def broad_policy(theta: float = 0.55, confirm: int = 1) -> Policy:
    return Policy(theta=float(theta), confirm_points=int(confirm), max_prob_surge=0.80,
                  max_price_runup=0.20, min_connection=0.40,
                  min_days_to_resolution=0.5, max_days_to_resolution=60.0,
                  max_event_assets=100, max_day_entries=100,
                  w_prob=1.0, w_connection=0.0, w_slope=0.0, w_runup=0.0)


def threshold_only_policy(cache: TradeCache, cutoff: pd.Timestamp):
    best = None
    for theta in THETA_GRID:
        for confirm in (1, 2, 3):
            p = broad_policy(float(theta), int(confirm))
            score, _ = evaluate_policy(cache, p, cutoff)
            if best is None or score > best[0]:
                best = (score, p)
    if best is None:
        raise RuntimeError("threshold-only grid produced no policy")
    return best[1], float(best[0])


def evaluate_policy_schedule(cache: TradeCache, schedule: dict[str, Policy], config: str) -> pd.DataFrame:
    parts = []
    for month in MONTHS:
        end = month + pd.offsets.MonthBegin(1)
        p = schedule[month.strftime("%Y-%m")]
        for delay in (0, 1):
            f = cache.get(p.theta, p.confirm_points, delay)
            x = f[(f.cross_ts >= month) & (f.cross_ts < end)]
            x = select_policy(x, p)
            if x.empty:
                continue
            x = x.copy(); x["configuration"] = config; x["benchmark"] = cache.benchmark
            x["eval_month"] = month.strftime("%Y-%m"); x["entry_delay"] = delay
            for k, v in asdict(p).items(): x[f"policy_{k}"] = v
            parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def candidate_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty: return {"n": 0}
    x = trades.excess_return.to_numpy(float)
    t = stats.ttest_1samp(x, 0.0)
    event = trades.groupby("event_group", dropna=False).excess_return.mean()
    return {
        "n": len(trades), "events": int(event.size),
        "mean_stock_pct": float(trades.stock_net_return.mean() * 100),
        "mean_benchmark_pct": float(trades.benchmark_net_return.mean() * 100),
        "mean_excess_pct": float(x.mean() * 100),
        "median_excess_pct": float(np.median(x) * 100),
        "win_vs_benchmark_pct": float((x > 0).mean() * 100),
        "t_stat": float(t.statistic), "p_two_sided": float(t.pvalue),
        "event_weighted_mean_excess_pct": float(event.mean() * 100),
    }


def cluster_bootstrap(trades: pd.DataFrame, cluster_col: str, rng: np.random.Generator, nboot: int = 5000) -> dict:
    g = trades.groupby(cluster_col, dropna=False).excess_return.apply(np.asarray)
    arrays = list(g); n = len(arrays); vals = np.empty(nboot)
    for i in range(nboot):
        picks = rng.integers(0, n, size=n)
        vals[i] = np.concatenate([arrays[j] for j in picks]).mean() * 100
    return {"cluster": cluster_col, "clusters": n,
            "ci_2_5_pct": float(np.quantile(vals, 0.025)),
            "median_pct": float(np.quantile(vals, 0.50)),
            "ci_97_5_pct": float(np.quantile(vals, 0.975)),
            "prob_mean_positive": float((vals > 0).mean())}


def portfolio_metrics(trades: pd.DataFrame, prices: dict[str, pd.DataFrame], benchmark: str) -> dict[str, Any]:
    b = prices[benchmark]
    dates = b.index[(b.index >= TRAIN_END.normalize()) & (b.index < TEST_END.normalize())]
    entries = {d: g.sort_values(["policy_score", "market_id", "symbol"], ascending=[False, True, True]) for d, g in trades.groupby("entry_date")}
    per_slot = START_CAPITAL / PORTFOLIO_SLOTS; b0 = float(b.loc[dates[0]].close); slots = []
    for _ in range(PORTFOLIO_SLOTS):
        buy = b0 * (1 + SLIPPAGE); q = per_slot / buy; fee = commission(q, buy); q = max((per_slot - fee) / buy, 0.0)
        slots.append({"kind": "benchmark", "qty": q})
    equity, realized = [], []; accepted = 0; skipped_capacity = 0
    for d in dates:
        bc = float(b.loc[d].close)
        for slot in slots:
            if slot["kind"] == "stock" and slot["exit_date"] == d:
                sell = slot["exit_close"] * (1 - SLIPPAGE); proceeds = slot["qty"] * sell
                fee = commission(slot["qty"], sell) + proceeds * SEC_FEE; cash = proceeds - fee
                realized.append(cash - slot["cash_out"])
                bbuy = bc * (1 + SLIPPAGE); bq = cash / bbuy; bfee = commission(bq, bbuy); bq = max((cash - bfee) / bbuy, 0.0)
                slot.clear(); slot.update({"kind": "benchmark", "qty": bq})
        if d in entries:
            for _, t in entries[d].iterrows():
                slot = next((s for s in slots if s["kind"] == "benchmark"), None)
                if slot is None: skipped_capacity += 1; continue
                bsell = bc * (1 - SLIPPAGE); cash = slot["qty"] * bsell
                fee = commission(slot["qty"], bsell) + cash * SEC_FEE; cash -= fee
                sbuy = float(t.entry_close) * (1 + SLIPPAGE); sq = cash / sbuy; sfee = commission(sq, sbuy); sq = max((cash - sfee) / sbuy, 0.0)
                spent = sq * sbuy + sfee
                slot.clear(); slot.update({"kind": "stock", "symbol": str(t.symbol), "qty": sq,
                                           "exit_date": pd.Timestamp(t.exit_date), "exit_close": float(t.exit_close), "cash_out": spent})
                accepted += 1
        mtm = 0.0
        for slot in slots:
            if slot["kind"] == "benchmark": mtm += slot["qty"] * bc
            else:
                sf = prices[slot["symbol"]]; i = sf.index.searchsorted(d, side="right") - 1
                if i >= 0: mtm += slot["qty"] * float(sf.iloc[i].close)
        equity.append(mtm)
    eq = np.asarray(equity, float); daily = np.diff(eq) / eq[:-1]
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1.0
    sr = float(eq[-1] / eq[0] - 1.0); br = float(b.loc[dates[-1]].close / b.loc[dates[0]].close - 1.0)
    return {"return_pct": sr * 100, "benchmark_return_pct": br * 100,
            "excess_return_pct": (sr - br) * 100, "sharpe": sharpe,
            "max_dd_pct": float(dd.min() * 100), "accepted_trades": accepted,
            "skipped_capacity": skipped_capacity,
            "win_rate_pct": float((np.asarray(realized) > 0).mean() * 100) if realized else np.nan,
            "realized_pnl": float(np.sum(realized))}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input-root", required=True); ap.add_argument("--output", required=True); args = ap.parse_args()
    inp = Path(args.input_root); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    policy_files = list(inp.rglob("selected_policies.csv")); trade_files = list(inp.rglob("oos_selected_trades.csv"))
    if not policy_files: raise RuntimeError(f"no seed policy files under {inp}")
    policies = pd.concat([pd.read_csv(p) for p in policy_files], ignore_index=True)
    seed_trades = pd.concat([pd.read_csv(p, parse_dates=["cross_ts", "entry_date", "exit_date", "t_e"]) for p in trade_files], ignore_index=True)
    for c in ["cross_ts", "entry_date", "exit_date", "t_e"]: seed_trades[c] = pd.to_datetime(seed_trades[c], utc=True)
    policies.to_csv(out / "all_seed_policies.csv", index=False); seed_trades.to_csv(out / "all_seed_oos_trades.csv", index=False)

    cand = pd.read_parquet(CANDIDATES); cand = cand[cand.cem_eligible.fillna(False).astype(bool)].copy()
    with open(PRICES, "rb") as f: raw_prices = pickle.load(f)
    with open(PROBS, "rb") as f: probs = pickle.load(f)
    prices = {k: make_price_frame(v) for k, v in raw_prices.items()}

    all_config_trades, schedule_records, threshold_records = [], [], []
    for benchmark in ("SPY", "QQQ"):
        cache = TradeCache(cand, prices, probs, benchmark)
        cons_schedule = {}; fixed_schedule = {m.strftime("%Y-%m"): broad_policy(0.55, 1) for m in MONTHS}; threshold_schedule = {}
        for month in MONTHS:
            key = month.strftime("%Y-%m"); g = policies[(policies.benchmark == benchmark) & (policies.month == key)]
            p = consensus_policy(g); cons_schedule[key] = p; schedule_records.append({"benchmark": benchmark, "month": key, **asdict(p)})
            tp, score = threshold_only_policy(cache, month); threshold_schedule[key] = tp; threshold_records.append({"benchmark": benchmark, "month": key, "objective_score": score, **asdict(tp)})
        for config, schedule in [("fixed_theta_055", fixed_schedule), ("threshold_only_walkforward", threshold_schedule), ("stochastic_consensus", cons_schedule)]:
            all_config_trades.append(evaluate_policy_schedule(cache, schedule, config))
    configs = pd.concat(all_config_trades, ignore_index=True)
    configs.to_csv(out / "consensus_and_control_oos_trades.csv", index=False)
    pd.DataFrame(schedule_records).to_csv(out / "consensus_policy_schedule.csv", index=False)
    pd.DataFrame(threshold_records).to_csv(out / "threshold_only_policy_schedule.csv", index=False)

    summary_rows, bootstrap_rows, portfolio_rows = [], [], []; rng = np.random.default_rng(20260716)
    for (benchmark, seed, delay), g in seed_trades.groupby(["benchmark", "seed", "entry_delay"]):
        summary_rows.append({"configuration": "stochastic_seed", "benchmark": benchmark, "seed": seed, "entry_delay": delay, **candidate_summary(g)})
    for (config, benchmark, delay), g in configs.groupby(["configuration", "benchmark", "entry_delay"]):
        summary_rows.append({"configuration": config, "benchmark": benchmark, "seed": "consensus", "entry_delay": delay, **candidate_summary(g)})
        gg = g.copy(); gg["entry_month"] = gg.entry_date.dt.to_period("M").astype(str)
        for col in ("event_group", "entry_month", "symbol"):
            bootstrap_rows.append({"configuration": config, "benchmark": benchmark, "entry_delay": delay, **cluster_bootstrap(gg, col, rng)})
        portfolio_rows.append({"configuration": config, "benchmark": benchmark, "entry_delay": delay, **portfolio_metrics(g, prices, benchmark)})

    summary = pd.DataFrame(summary_rows); bootstrap = pd.DataFrame(bootstrap_rows); portfolios = pd.DataFrame(portfolio_rows)
    summary.to_csv(out / "candidate_oos_summary.csv", index=False); bootstrap.to_csv(out / "cluster_bootstrap.csv", index=False); portfolios.to_csv(out / "portfolio_oos_summary.csv", index=False)
    monthly = configs.assign(entry_month=configs.entry_date.dt.to_period("M").astype(str)).groupby(["configuration", "benchmark", "entry_delay", "entry_month"]).agg(n=("excess_return", "size"), mean_excess_pct=("excess_return", lambda x: x.mean() * 100), net_stock_return_sum_pct=("stock_net_return", lambda x: x.sum() * 100)).reset_index()
    monthly.to_csv(out / "monthly_oos_diagnostics.csv", index=False)
    param_cols = ["theta", "confirm_points", "max_prob_surge", "max_price_runup", "min_connection", "min_days_to_resolution", "max_days_to_resolution", "max_event_assets", "max_day_entries", "w_prob", "w_connection", "w_slope", "w_runup"]
    stability = policies.groupby(["benchmark", "month"])[param_cols].agg(["median", "std", "min", "max"]); stability.columns = [f"{a}_{b}" for a, b in stability.columns]
    stability.reset_index().to_csv(out / "optimizer_parameter_stability.csv", index=False)
    lines = ["# Stochastic fixed-horizon optimization", "", "- Entry is recomputed from the effective probability path for every theta.", "- Exit is always Te-1.", "- Five optimizer seeds; no best-seed selection.", "- Monthly outer walk-forward; only completed outcomes enter training.", "- Objective is robust to same-close and next-close entry.", "", "## Candidate OOS", "", summary.to_markdown(index=False), "", "## Portfolio OOS", "", portfolios.to_markdown(index=False), "", "## Cluster bootstrap", "", bootstrap.to_markdown(index=False)]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False)); print("\nPORTFOLIOS\n", portfolios.to_string(index=False))


if __name__ == "__main__":
    main()
