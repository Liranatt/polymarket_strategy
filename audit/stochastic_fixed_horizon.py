from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.polarity import effective_prob_path, resolve_polarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_ROOT = ROOT / "runs"

CANDIDATES = DATA / "candidates_audit_clean.parquet"
PRICES = DATA / "prices.pkl"
PROBS = DATA / "probs.pkl"

TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-07-01", tz="UTC")
MONTHS = pd.date_range(TRAIN_END, TEST_END - pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
THETA_GRID = np.round(np.arange(0.55, 0.901, 0.005), 3)
CONFIRM_VALUES = (1, 2, 3)
ENTRY_DELAYS = (0, 1)

START_CAPITAL = 100_000.0
PORTFOLIO_SLOTS = 10
SLIPPAGE = 0.0005
SEC_FEE = 0.0000278


def utc(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def commission(shares: float, price: float) -> float:
    tv = shares * price
    return max(0.35, min(shares * 0.0035, tv * 0.01))


def net_return(entry: float, exit_: float, notional: float = 10_000.0) -> float:
    buy = entry * (1.0 + SLIPPAGE)
    sell = exit_ * (1.0 - SLIPPAGE)
    shares = notional / buy
    costs = commission(shares, buy) + commission(shares, sell) + shares * sell * SEC_FEE
    return (shares * sell - costs) / notional - 1.0


def make_price_frame(rows: list[tuple]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([utc(r[0]).normalize() for r in rows])
    return pd.DataFrame(
        {"high": [float(r[1]) for r in rows], "low": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows]},
        index=idx,
    ).sort_index()


def first_close_on_or_after(frame: pd.DataFrame, ts: pd.Timestamp, delay: int = 0):
    i = frame.index.searchsorted(ts.normalize(), side="left") + int(delay)
    if i >= len(frame):
        return None
    return frame.index[i], float(frame.iloc[i].close)


def last_close_before(frame: pd.DataFrame, ts: pd.Timestamp):
    i = frame.index.searchsorted(ts.normalize(), side="left") - 1
    if i < 0:
        return None
    return frame.index[i], float(frame.iloc[i].close)


def last_prob_at_or_before(points: list[tuple[pd.Timestamp, float]], ts: pd.Timestamp) -> float | None:
    if not points:
        return None
    vals = np.array([p[0].value for p in points], dtype=np.int64)
    i = np.searchsorted(vals, ts.value, side="right") - 1
    if i < 0:
        return float(points[0][1])
    return float(points[i][1])


def daily_last_points(points: list[tuple]) -> list[tuple[pd.Timestamp, float]]:
    by_day: dict[pd.Timestamp, tuple[pd.Timestamp, float]] = {}
    for raw_t, raw_v in points:
        t = utc(raw_t)
        by_day[t.normalize()] = (t, float(raw_v))
    return [by_day[k] for k in sorted(by_day)]


def find_confirmed_crossing(points, first_eligible, theta, confirm_points):
    held = 0
    for i, (t, v) in enumerate(points):
        if t < first_eligible:
            continue
        if v >= theta:
            held += 1
            if held >= confirm_points:
                return i, t, v
        else:
            held = 0
    return None


@dataclass(frozen=True)
class Policy:
    theta: float
    confirm_points: int
    max_prob_surge: float
    max_price_runup: float
    min_connection: float
    min_days_to_resolution: float
    max_days_to_resolution: float
    max_event_assets: int
    max_day_entries: int
    w_prob: float
    w_connection: float
    w_slope: float
    w_runup: float


class TradeCache:
    def __init__(self, candidates: pd.DataFrame, prices: dict, probs: dict, benchmark: str):
        self.candidates = candidates.copy()
        self.prices = prices
        self.probs = probs
        self.benchmark = benchmark
        self._cache: dict[tuple[float, int, int], pd.DataFrame] = {}
        self._base = self._prepare_base()

    def _prepare_base(self) -> list[dict[str, Any]]:
        base = []
        for row in self.candidates.itertuples(index=False):
            symbol = str(row.symbol)
            market_id = str(row.market_id)
            if symbol not in self.prices or self.benchmark not in self.prices:
                continue
            polarity, source = resolve_polarity(str(row.question), symbol)
            if polarity == 0:
                continue
            raw_path = self.probs.get(market_id, [])
            if not raw_path:
                continue
            path = daily_last_points(effective_prob_path(raw_path, polarity))
            if not path:
                continue
            t_theta = utc(row.t_theta)
            t0 = utc(row.t0)
            te = utc(row.t_e)
            screen_prob = last_prob_at_or_before(path, t_theta)
            if screen_prob is None:
                continue
            t0_close = first_close_on_or_after(self.prices[symbol], t0, 0)
            if t0_close is None:
                continue
            base.append({
                "event_id": str(row.event_id), "market_id": market_id, "symbol": symbol,
                "question": str(row.question), "split": str(row.split), "t0": t0,
                "stored_t_theta": t_theta, "t_e": te, "screen_prob": float(screen_prob),
                "path": path,
                "connection": float(row.feat_connection_strength) if pd.notna(row.feat_connection_strength) else 0.0,
                "world_size": int(row.feat_world_size) if pd.notna(row.feat_world_size) else 1,
                "event_group": str(row.economic_event_id) if pd.notna(row.economic_event_id) else str(row.event_id),
                "event_type": str(row.current_family or row.feat_archetype or "other"),
                "t0_close_date": t0_close[0], "t0_close": t0_close[1],
                "polarity": polarity, "polarity_source": source,
            })
        return base

    def get(self, theta: float, confirm_points: int, entry_delay: int) -> pd.DataFrame:
        key = (round(float(theta), 3), int(confirm_points), int(entry_delay))
        if key in self._cache:
            return self._cache[key]
        out = []
        bpx = self.prices[self.benchmark]
        for rec in self._base:
            crossing = find_confirmed_crossing(rec["path"], rec["stored_t_theta"], key[0], key[1])
            if crossing is None:
                continue
            _, cross_ts, entry_prob = crossing
            ent = first_close_on_or_after(self.prices[rec["symbol"]], cross_ts, key[2])
            ex = last_close_before(self.prices[rec["symbol"]], rec["t_e"])
            if ent is None or ex is None or ex[0] < ent[0]:
                continue
            bent = first_close_on_or_after(bpx, ent[0], 0)
            bex = last_close_before(bpx, rec["t_e"])
            if bent is None or bex is None or bex[0] < bent[0]:
                continue
            prior_24 = last_prob_at_or_before(rec["path"], cross_ts - pd.Timedelta(hours=24))
            slope_24 = float(entry_prob - prior_24) if prior_24 is not None else 0.0
            prob_surge = float(entry_prob - rec["screen_prob"])
            price_runup = float(ent[1] / rec["t0_close"] - 1.0)
            days_to_resolution = float((rec["t_e"] - cross_ts).total_seconds() / 86400.0)
            stock_ret = net_return(ent[1], ex[1])
            benchmark_ret = net_return(bent[1], bex[1])
            out.append({
                **{k: v for k, v in rec.items() if k != "path"},
                "theta": key[0], "confirm_points": key[1], "entry_delay": key[2],
                "cross_ts": cross_ts, "entry_date": ent[0], "exit_date": ex[0],
                "entry_close": ent[1], "exit_close": ex[1],
                "benchmark_entry_close": bent[1], "benchmark_exit_close": bex[1],
                "entry_prob": float(entry_prob), "prob_surge_from_screen": prob_surge,
                "price_runup_from_t0": price_runup, "prob_slope_24h": slope_24,
                "days_to_resolution": days_to_resolution, "stock_net_return": stock_ret,
                "benchmark_net_return": benchmark_ret, "excess_return": stock_ret - benchmark_ret,
            })
        frame = pd.DataFrame(out)
        self._cache[key] = frame
        return frame


def soft_rank_score(frame: pd.DataFrame, p: Policy) -> pd.Series:
    prob = ((frame.entry_prob - 0.55) / 0.35).clip(0, 1)
    conn = ((frame.connection - 0.4) / 0.6).clip(0, 1)
    slope = ((frame.prob_slope_24h + 0.10) / 0.60).clip(0, 1)
    runup = ((frame.price_runup_from_t0 + 0.10) / 0.30).clip(0, 1)
    return p.w_prob * prob + p.w_connection * conn + p.w_slope * slope - p.w_runup * runup


def select_policy(frame: pd.DataFrame, p: Policy) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        (frame.prob_surge_from_screen <= p.max_prob_surge)
        & (frame.price_runup_from_t0 <= p.max_price_runup)
        & (frame.connection >= p.min_connection)
        & (frame.days_to_resolution >= p.min_days_to_resolution)
        & (frame.days_to_resolution <= p.max_days_to_resolution)
    )
    x = frame.loc[mask].copy()
    if x.empty:
        return x
    x["policy_score"] = soft_rank_score(x, p)
    x = x.sort_values(["entry_date", "event_group", "policy_score", "market_id", "symbol"], ascending=[True, True, False, True, True])
    x = x.groupby(["entry_date", "event_group"], group_keys=False).head(p.max_event_assets)
    x = x.sort_values(["entry_date", "policy_score", "market_id", "symbol"], ascending=[True, False, True, True])
    x = x.groupby("entry_date", group_keys=False).head(p.max_day_entries)
    return x.reset_index(drop=True)


def robust_score(selected: pd.DataFrame) -> float:
    n = len(selected)
    if n < 60:
        return -1e9
    events = selected.groupby("event_group", dropna=False).excess_return.mean()
    months = selected.assign(month=selected.entry_date.dt.to_period("M").astype(str)).groupby("month").excess_return.mean()
    if len(events) < 20 or len(months) < 6:
        return -1e9
    event_mean = float(events.mean())
    event_se = float(events.std(ddof=1) / math.sqrt(len(events))) if len(events) > 1 else 1.0
    month_median = float(months.median())
    month_std = float(months.std(ddof=1)) if len(months) > 1 else 1.0
    positive_month_share = float((months > 0).mean())
    abs_contrib = selected.groupby("event_group").excess_return.sum().abs().sort_values(ascending=False)
    top_share = float(abs_contrib.head(5).sum() / abs_contrib.sum()) if abs_contrib.sum() > 0 else 1.0
    concentration_penalty = max(0.0, top_share - 0.35)
    return event_mean - 0.75 * event_se + 0.35 * month_median - 0.20 * month_std + 0.005 * (positive_month_share - 0.5) - 0.02 * concentration_penalty


def evaluate_policy(cache: TradeCache, p: Policy, history_cutoff: pd.Timestamp):
    scores = []
    info = {}
    for delay in ENTRY_DELAYS:
        f = cache.get(p.theta, p.confirm_points, delay)
        h = f[f.exit_date < history_cutoff]
        s = select_policy(h, p)
        score = robust_score(s)
        scores.append(score)
        info[f"score_delay_{delay}"] = score
        info[f"n_delay_{delay}"] = float(len(s))
    if min(scores) <= -1e8:
        return -1e9, info
    final = 0.70 * min(scores) + 0.30 * float(np.mean(scores))
    info["robust_score"] = final
    return final, info


class Sampler:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.mu = np.array([0.725, 0.45, 0.10, 0.70, 3.0, 35.0])
        self.sd = np.array([0.10, 0.20, 0.06, 0.18, 2.0, 14.0])
        self.confirm_p = np.array([1/3, 1/3, 1/3])
        self.event_p = np.array([0.35, 0.30, 0.22, 0.13])
        self.day_values = np.array([3, 5, 8, 10, 15])
        self.day_p = np.ones(len(self.day_values)) / len(self.day_values)
        self.weight_alpha = np.ones(4)

    def sample(self, n: int) -> list[Policy]:
        raw = self.rng.normal(self.mu, self.sd, size=(n, len(self.mu)))
        raw[:, 0] = np.clip(raw[:, 0], 0.55, 0.90)
        raw[:, 1] = np.clip(raw[:, 1], 0.10, 0.80)
        raw[:, 2] = np.clip(raw[:, 2], 0.00, 0.20)
        raw[:, 3] = np.clip(raw[:, 3], 0.40, 1.00)
        raw[:, 4] = np.clip(raw[:, 4], 0.5, 10.0)
        raw[:, 5] = np.clip(raw[:, 5], 7.0, 60.0)
        confirms = self.rng.choice(CONFIRM_VALUES, size=n, p=self.confirm_p)
        events = self.rng.choice([1, 2, 3, 4], size=n, p=self.event_p)
        days = self.rng.choice(self.day_values, size=n, p=self.day_p)
        weights = self.rng.dirichlet(self.weight_alpha, size=n)
        out = []
        for i in range(n):
            min_d = float(raw[i, 4])
            max_d = max(float(raw[i, 5]), min_d + 3.0)
            theta = float(THETA_GRID[np.abs(THETA_GRID - raw[i, 0]).argmin()])
            out.append(Policy(
                theta=theta, confirm_points=int(confirms[i]),
                max_prob_surge=float(raw[i, 1]), max_price_runup=float(raw[i, 2]),
                min_connection=float(raw[i, 3]), min_days_to_resolution=min_d,
                max_days_to_resolution=max_d, max_event_assets=int(events[i]),
                max_day_entries=int(days[i]), w_prob=float(weights[i, 0]),
                w_connection=float(weights[i, 1]), w_slope=float(weights[i, 2]),
                w_runup=float(weights[i, 3]),
            ))
        return out

    def update(self, elite: list[Policy]) -> None:
        arr = np.array([[p.theta, p.max_prob_surge, p.max_price_runup, p.min_connection, p.min_days_to_resolution, p.max_days_to_resolution] for p in elite])
        self.mu = 0.25 * self.mu + 0.75 * arr.mean(axis=0)
        self.sd = np.maximum(0.25 * self.sd + 0.75 * arr.std(axis=0, ddof=0), np.array([0.005, 0.02, 0.01, 0.02, 0.25, 1.0]))
        self.confirm_p = smoothed_freq([p.confirm_points for p in elite], [1, 2, 3])
        self.event_p = smoothed_freq([p.max_event_assets for p in elite], [1, 2, 3, 4])
        self.day_p = smoothed_freq([p.max_day_entries for p in elite], self.day_values.tolist())
        w = np.array([[p.w_prob, p.w_connection, p.w_slope, p.w_runup] for p in elite])
        self.weight_alpha = 1.0 + 20.0 * w.mean(axis=0)


def smoothed_freq(values: list[int], support: list[int]) -> np.ndarray:
    counts = np.ones(len(support), dtype=float)
    for v in values:
        counts[support.index(v)] += 1.0
    return counts / counts.sum()


def optimize(cache: TradeCache, cutoff: pd.Timestamp, seed: int, iterations: int, population: int):
    sampler = Sampler(seed)
    trace = []
    best = None
    for iteration in range(iterations):
        policies = sampler.sample(population)
        evaluated = []
        for p in policies:
            score, info = evaluate_policy(cache, p, cutoff)
            evaluated.append((score, p, info))
            trace.append({"iteration": iteration, "score": score, **asdict(p), **info})
            if best is None or score > best[0]:
                best = (score, p, info)
        evaluated.sort(key=lambda z: z[0], reverse=True)
        valid = [x for x in evaluated if x[0] > -1e8]
        if valid:
            elite_n = max(10, int(math.ceil(0.15 * len(valid))))
            sampler.update([x[1] for x in valid[:elite_n]])
    if best is None or best[0] <= -1e8:
        raise RuntimeError(f"no valid policy for {cache.benchmark} cutoff {cutoff}")
    return best, pd.DataFrame(trace)


def month_metrics(selected, benchmark, seed, month, delay, p):
    x = selected.excess_return.to_numpy(float) if not selected.empty else np.array([])
    return {
        "benchmark": benchmark, "seed": seed, "month": month.strftime("%Y-%m"),
        "entry_delay": delay, "n": len(selected),
        "mean_stock_pct": float(selected.stock_net_return.mean() * 100) if len(selected) else np.nan,
        "mean_excess_pct": float(x.mean() * 100) if len(x) else np.nan,
        "median_excess_pct": float(np.median(x) * 100) if len(x) else np.nan,
        "win_vs_benchmark_pct": float((x > 0).mean() * 100) if len(x) else np.nan,
        **{f"policy_{k}": v for k, v in asdict(p).items()},
    }


def evaluate_month(cache, p, start, end, seed):
    rows, metrics = [], []
    for delay in ENTRY_DELAYS:
        f = cache.get(p.theta, p.confirm_points, delay)
        month = f[(f.cross_ts >= start) & (f.cross_ts < end)]
        selected = select_policy(month, p)
        if not selected.empty:
            selected = selected.copy()
            selected["benchmark"] = cache.benchmark
            selected["seed"] = seed
            selected["eval_month"] = start.strftime("%Y-%m")
            selected["entry_delay"] = delay
            for k, v in asdict(p).items():
                selected[f"policy_{k}"] = v
            rows.extend(selected.to_dict("records"))
        metrics.append(month_metrics(selected, cache.benchmark, seed, start, delay, p))
    return rows, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--population", type=int, default=120)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    out = OUT_ROOT / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    cand = pd.read_parquet(CANDIDATES)
    cand = cand[cand.cem_eligible.fillna(False).astype(bool)].copy()
    forbidden = {"asset_return", "expected_return_pct", "confidence_score", "feat_llm_expected_return", "feat_llm_confidence"}
    assert forbidden.issubset(set(cand.columns))
    with open(PRICES, "rb") as f:
        raw_prices = pickle.load(f)
    with open(PROBS, "rb") as f:
        probs = pickle.load(f)
    prices = {k: make_price_frame(v) for k, v in raw_prices.items()}

    all_policies, all_selected, all_month_metrics, trace_parts = [], [], [], []
    for benchmark in ("SPY", "QQQ"):
        cache = TradeCache(cand, prices, probs, benchmark)
        parity = cache.get(0.55, 1, 0)
        if not parity.empty:
            delta = (parity.cross_ts - parity.stored_t_theta).dt.total_seconds() / 86400.0
            pd.DataFrame({"delta_days": delta}).to_csv(out / f"entry_recompute_parity_{benchmark.lower()}.csv", index=False)
        for month_idx, month in enumerate(MONTHS):
            end = month + pd.offsets.MonthBegin(1)
            fit_seed = args.seed + (10_000 if benchmark == "QQQ" else 0) + month_idx * 1_000
            best, trace = optimize(cache, month, fit_seed, args.iterations, args.population)
            score, policy, info = best
            all_policies.append({"benchmark": benchmark, "seed": args.seed, "month": month.strftime("%Y-%m"), "fit_seed": fit_seed, "objective_score": score, **asdict(policy), **info})
            trace["benchmark"] = benchmark
            trace["seed"] = args.seed
            trace["month"] = month.strftime("%Y-%m")
            trace_parts.append(trace)
            rows, metrics = evaluate_month(cache, policy, month, end, args.seed)
            all_selected.extend(rows)
            all_month_metrics.extend(metrics)
            print(benchmark, month.strftime("%Y-%m"), "score", round(score, 6), json.dumps(asdict(policy), sort_keys=True), flush=True)

    pd.DataFrame(all_policies).to_csv(out / "selected_policies.csv", index=False)
    pd.DataFrame(all_selected).to_csv(out / "oos_selected_trades.csv", index=False)
    pd.DataFrame(all_month_metrics).to_csv(out / "oos_monthly_metrics.csv", index=False)
    trace_all = pd.concat(trace_parts, ignore_index=True)
    trace_all.to_csv(out / "search_trace.csv.gz", index=False, compression="gzip")
    manifest = {
        "seed": args.seed, "iterations": args.iterations, "population": args.population,
        "policy_evaluations": int(len(trace_all)),
        "entry_semantics": "recomputed effective-probability crossing; Te-1 exit",
        "objective": "70% worst-delay robust score + 30% mean across same-close and next-close entry",
        "forbidden_outcome_features": sorted(forbidden),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
