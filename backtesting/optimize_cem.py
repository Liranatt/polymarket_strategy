"""
Leakage-safe 8-experiment comparison:
  T1 = realized-friction penalty inside the CEM objective
       (this is NOT a live entry gate because this non-RF runner has no
        point-in-time expected-return estimate);
  T2 = expanding walk-forward CEM folds:
       each fold fits only on rows whose outcome was complete before that
       fold starts (t_e < fold start), then evaluates the next time block;
  T3 = half-Kelly position sizing from realised, fully net historical trades.

Key corrections relative to the original runner:
  * CEM never reads the test split.
  * Validation candidates are evaluated after training and before the test
    boundary. They can be used to choose an experiment family; test remains the
    untouched final holdout.
  * T2 uses genuine expanding walk-forward folds. Fold i fits on every
    train-split row with t_e < fold_i_start, then evaluates the next block
    of candidates by t_theta. A candidate's t_theta alone never makes it
    eligible for fitting.
  * Every fit/evaluation simulation is valuated at its own cutoff date; it
    cannot use prices after that cutoff.
  * Finite-horizon simulations truncate price/probability paths before trade
    generation, so entry/exit decisions cannot inspect later observations.
  * OOS metrics come from a separate, frozen-policy portfolio simulation on
    test candidates only, with one fixed OOS end date across all ablations.
    They are actual portfolio returns, not summed trade P&L.
  * Trade P&L is fully net of all modeled rotation costs:
      benchmark sell + asset buy + asset sell + benchmark rebuy.
  * The CEM objective uses daily portfolio-equity Sharpe, not a small sample
    of trade returns.
  * Each experiment starts CEM from the same benchmark-specific random seed,
    so an ablation is not confounded by a different initial population.
  * Kelly uses fully net realised returns and reports actual realised sizing.

Usage:
    python -u run_experiments.py

This file deliberately writes new output names so the old contaminated result
CSV is preserved:
    data/experiment_results_clean.csv
    data/experiment_trade_logs_clean/
    data/experiment_equity_logs_clean/
    data/experiment_walkforward_folds_clean.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):  # absent under Jupyter's OutStream
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.policy import DEFAULT_POLICY
from core.kernel import (
    clear_kernel_caches,
    simulate_one,
    entry_day,
)
from core.polarity import effective_prob_path, effective_prob_surge, resolve_polarity
from backtesting.pipeline.trade_forensics import combine_forensic_csvs, write_trade_forensics


PROJECT = Path(__file__).resolve().parent.parent
REL_COL = "feat_connection_strength"

RESULTS_CSV = PROJECT / "data" / "experiment_results_clean.csv"
TRADE_LOG_DIR = PROJECT / "data" / "experiment_trade_logs_clean"
EQUITY_LOG_DIR = PROJECT / "data" / "experiment_equity_logs_clean"
ALLOCATION_LOG_DIR = PROJECT / "data" / "experiment_allocation_logs_clean"
FORENSIC_LOG_DIR = PROJECT / "data" / "experiment_forensics_clean"
WF_FOLD_AUDIT_CSV = PROJECT / "data" / "experiment_walkforward_folds_clean.csv"
DISPOSITION_DIR = PROJECT / "output"
CEM_POPULATION_CSV = PROJECT / "output" / "cem_population.csv"
DEFAULT_CANDIDATES_PATH = PROJECT / "data" / "candidates_audit_clean.parquet"


@dataclass(frozen=True)
class OutputPaths:
    """Every file a run writes. Namespaced by `run_id` so concurrent or repeated
    runs (e.g. a seed sweep) cannot clobber each other's audit trail."""

    results_csv: Path
    trade_log_dir: Path
    equity_log_dir: Path
    allocation_log_dir: Path
    forensic_log_dir: Path
    disposition_dir: Path
    wf_fold_audit_csv: Path
    cem_population_csv: Path
    plot_dir: Path


@dataclass(frozen=True)
class DynamicPolicySchedule:
    """Callable walk-forward policy schedule for online-refit arms."""

    windows: list[dict[str, Any]]

    def __call__(self, day: pd.Timestamp) -> dict:
        day = as_utc_day(day)
        if not self.windows:
            return {}
        matched = self.windows[0]["policy"]
        for window in self.windows:
            start = window["eval_start"]
            end_excl = window["eval_end_exclusive"]
            policy = window["policy"]
            if day >= start and day < end_excl:
                return policy
            if day >= start:
                matched = policy
        return matched

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "fold": int(window["fold"]),
                "eval_start_date": str(window["eval_start"].date()),
                "eval_end_exclusive_date": str(window["eval_end_exclusive"].date()),
                "policy": window["policy"],
            }
            for window in self.windows
        ]


def resolve_output_paths(run_id: str | None = None) -> OutputPaths:
    """`run_id=None` reproduces the historical, un-namespaced layout exactly."""
    if not run_id:
        return OutputPaths(
            results_csv=RESULTS_CSV,
            trade_log_dir=TRADE_LOG_DIR,
            equity_log_dir=EQUITY_LOG_DIR,
            allocation_log_dir=ALLOCATION_LOG_DIR,
            forensic_log_dir=FORENSIC_LOG_DIR,
            disposition_dir=DISPOSITION_DIR,
            wf_fold_audit_csv=WF_FOLD_AUDIT_CSV,
            cem_population_csv=CEM_POPULATION_CSV,
            plot_dir=RESULTS_CSV.parent,
        )
    root = PROJECT / "runs" / _slug(run_id)
    return OutputPaths(
        results_csv=root / "experiment_results_clean.csv",
        trade_log_dir=root / "experiment_trade_logs_clean",
        equity_log_dir=root / "experiment_equity_logs_clean",
        allocation_log_dir=root / "experiment_allocation_logs_clean",
        forensic_log_dir=root / "experiment_forensics_clean",
        disposition_dir=root / "dispositions",
        wf_fold_audit_csv=root / "experiment_walkforward_folds_clean.csv",
        cem_population_csv=root / "cem_population.csv",
        plot_dir=root / "plots",
    )


INITIAL_CAPITAL = 100_000.0

# Fully-invested constraint: idle cash is swept into the benchmark daily, so
# capital is always either in an event position or in the index. Rotation
# residuals no longer accumulate as cash drag.
FULLY_INVESTED_SWEEP = True

# SPY/QQQ are fraction-eligible at Interactive Brokers, so benchmark rotation
# legs may use fractional shares. Event positions stay whole-share.
FRACTIONAL_BENCHMARK = True
# Don't churn sub-$100 sweeps: the $0.35 minimum commission would exceed 35 bp.
MIN_SWEEP_CASH = 100.0

# ── Portfolio parameter space ────────────────────────────────────────────────

PORTFOLIO_BOUNDS = dict(
    atr_mult=(1.5, 4.0),
    lock_activate=(0.02, 0.10),
    theta_out=(0.45, 0.60),
    enter_strong=(0.60, 0.85),
    enter_floor=(0.55, 0.80),
    hold_days=(1, 5),
    max_prob_surge=(0.20, 0.80),
    max_price_runup=(0.02, 0.20),
    position_size_pct=(0.06, 0.12),
    max_concurrent=(8, 12),
)
PORT_DEFAULT = {**DEFAULT_POLICY, "position_size_pct": 0.10, "max_concurrent": 10}

# ── Experiment controls ──────────────────────────────────────────────────────

HURDLE_MULT = 3.0
HURDLE_PENALTY = 2.0
# T1 is a realised friction penalty used only in the CEM fitness. A true,
# live pre-entry hurdle needs a point-in-time expected-return estimate.

WF_EVAL_MON = 3
WF_STEP_MON = 3
WF_MIN_TRAIN_CANDS = 8
WF_MIN_EVAL_CANDS = 8
WF_MIN_FOLDS = 2

# Every CEM fit is clipped at the next fold/stage boundary. The t_e rule decides
# whether a row is eligible to fit; the horizon rule prevents path-level leaks.

KELLY_MIN_N = 10
KELLY_LOOKBACK_N = 30
KELLY_MIN_SZ = 0.05
KELLY_MAX_SZ = 0.2

CEM_ITERS = 6
CEM_POP = 20
CEM_ELITE_FRAC = 0.25
CEM_BASE_SEED = 42
# Every ablation gets the identical initial CEM population for a given benchmark.
# SPY and QQQ have separate but fixed populations.
BENCHMARK_SEED_OFFSET = {"SPY": 0, "QQQ": 10_000}

MIN_TRADES_FOR_REWARD = 3
MIN_DAILY_RETURNS_FOR_SHARPE = 20
DD_PENALTY = 0.30
INVALID_SCORE = -1e9
EVAL_TRADE_WARNING_N = 80

ALLOCATION_FIFO = "fifo"
ALLOCATION_EVENT_PRIORITY = "event_priority"
EVENT_PRIORITY_ORDER = {"geo": 0, "macro": 1, "earnings": 2, "other": 3}
RANK_RUNUP_CLIP = (-0.20, 0.20)
PREEMPT_NET_PROFIT_HURDLE_PCT = 3.0

EXPERIMENTS = [
    {"id": 0, "label": "Baseline", "hurdle": False, "wf": False, "kelly": False},
    # Standalone single-technique arms and the T*+T3 pairs are parked to cut the
    # matrix from 20 cells to 10 while the seed-variance study runs. Restore them
    # once the noise floor is known. `id` is the stable key -- nothing depends on
    # this list's ordinal position.
    # {"id": 1, "label": "T1 FrictionPenalty", "hurdle": True, "wf": False, "kelly": False},
    # {"id": 2, "label": "T2 TrainWindows", "hurdle": False, "wf": True, "kelly": False},
    # {"id": 3, "label": "T3 Kelly", "hurdle": False, "wf": False, "kelly": True},
    {"id": 4, "label": "T1+T2", "hurdle": True, "wf": True, "kelly": False},
    # {"id": 5, "label": "T1+T3", "hurdle": True, "wf": False, "kelly": True},
    # {"id": 6, "label": "T2+T3", "hurdle": False, "wf": True, "kelly": True},
    {"id": 7, "label": "T1+T2+T3", "hurdle": True, "wf": True, "kelly": True},
    {"id": 8, "label": "T4 GeoPriority", "hurdle": False, "wf": False, "kelly": False, "allocation_mode": ALLOCATION_EVENT_PRIORITY},
    {"id": 9, "label": "T1+T2+T3+T4", "hurdle": True, "wf": True, "kelly": True, "allocation_mode": ALLOCATION_EVENT_PRIORITY},
]

# Every arm ever defined, keyed by CLI slug. `--experiments` selects from this map,
# so a parked arm can still be run explicitly without editing source.
ALL_EXPERIMENTS = [
    {"id": 0, "label": "Baseline", "hurdle": False, "wf": False, "kelly": False},
    {"id": 1, "label": "T1 FrictionPenalty", "hurdle": True, "wf": False, "kelly": False},
    {"id": 2, "label": "T2 TrainWindows", "hurdle": False, "wf": True, "kelly": False},
    {"id": 3, "label": "T3 Kelly", "hurdle": False, "wf": False, "kelly": True},
    {"id": 4, "label": "T1+T2", "hurdle": True, "wf": True, "kelly": False},
    {"id": 5, "label": "T1+T3", "hurdle": True, "wf": False, "kelly": True},
    {"id": 6, "label": "T2+T3", "hurdle": False, "wf": True, "kelly": True},
    {"id": 7, "label": "T1+T2+T3", "hurdle": True, "wf": True, "kelly": True},
    {"id": 8, "label": "T4 GeoPriority", "hurdle": False, "wf": False, "kelly": False, "allocation_mode": ALLOCATION_EVENT_PRIORITY},
    {"id": 9, "label": "T1+T2+T3+T4", "hurdle": True, "wf": True, "kelly": True, "allocation_mode": ALLOCATION_EVENT_PRIORITY},
]


# ── Time / price helpers ─────────────────────────────────────────────────────

_CLOSE_CACHE: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
_PATH_CUTOFF_CACHE: dict[
    tuple[int, int, str],
    tuple[dict[str, list[tuple]], dict[str, list[tuple]]],
] = {}

_DAY_NS = 86_400_000_000_000


def as_utc_day(value: Any) -> pd.Timestamp:
    """Convert a date-like value to a normalized UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def ib_cost(shares: int, price: float, is_sell: bool) -> float:
    """IB-style commission + SEC fee on sales + fixed 5 bp slippage."""
    if shares <= 0 or price <= 0:
        return 0.0
    trade_value = shares * price
    commission = max(0.35, min(shares * 0.0035, trade_value * 0.01))
    sec = trade_value * 0.0000278 if is_sell else 0.0
    return commission + sec + trade_value * 0.0005


def _close_on(prices: dict, symbol: str, date: Any) -> float | None:
    """Latest known daily close on or before date, cached for CEM speed.

    The cache holds int64 day-ns and float closes so the per-call lookup is a
    plain ``np.searchsorted`` with no pandas object overhead. Already-normalized
    UTC timestamps (every call site in the daily loop) skip ``as_utc_day``.
    """
    key = (id(prices), symbol)
    cached = _CLOSE_CACHE.get(key)
    if cached is None:
        bars = prices.get(symbol, [])
        if not bars:
            return None
        days = np.array([as_utc_day(t).value for t, *_ in bars], dtype=np.int64)
        values = np.asarray([float(close) for *_rest, close in bars], dtype=float)
        cached = (days, values)
        _CLOSE_CACHE[key] = cached

    days, values = cached
    if type(date) is pd.Timestamp and date.tzinfo is not None and (date.value % _DAY_NS) == 0:
        date_ns = date.value
    else:
        date_ns = as_utc_day(date).value
    loc = int(np.searchsorted(days, date_ns, side="right")) - 1
    if loc < 0:
        return None
    return float(values[loc])


def _calendar_dates(prices: dict, bench_sym: str, start_date: Any, end_date: Any) -> list[pd.Timestamp]:
    start = as_utc_day(start_date)
    end = as_utc_day(end_date)
    dates = sorted({as_utc_day(t) for t, *_ in prices.get(bench_sym, [])})
    return [d for d in dates if start <= d <= end]


def truncate_paths(prices: dict, probs: dict, end_date: Any | None) -> tuple[dict, dict]:
    """Return price/probability paths clipped to an evaluation horizon."""
    if end_date is None:
        return prices, probs

    cutoff = as_utc_day(end_date)
    key = (id(prices), id(probs), str(cutoff.date()))
    cached = _PATH_CUTOFF_CACHE.get(key)
    if cached is not None:
        return cached

    truncated_prices = {
        symbol: [bar for bar in bars if as_utc_day(bar[0]) <= cutoff]
        for symbol, bars in prices.items()
    }
    truncated_probs = {
        market_id: [point for point in points if as_utc_day(point[0]) <= cutoff]
        for market_id, points in probs.items()
    }
    cached = (truncated_prices, truncated_probs)
    _PATH_CUTOFF_CACHE[key] = cached
    return cached


def _affordable_buy_qty(cash_available: float, price: float) -> int:
    """Largest integer share count whose buy price plus modeled buy cost fits cash."""
    if cash_available <= 0 or price <= 0:
        return 0
    qty = int(cash_available / price)
    while qty > 0 and qty * price + ib_cost(qty, price, False) > cash_available + 1e-9:
        qty -= 1
    return qty


def _bench_buy_qty(cash_available: float, price: float) -> float:
    """Benchmark buy size: fractional shares when enabled, else whole shares."""
    if not FRACTIONAL_BENCHMARK:
        return float(_affordable_buy_qty(cash_available, price))
    if cash_available <= 0 or price <= 0:
        return 0.0
    qty = cash_available / price
    for _ in range(4):
        qty = max((cash_available - ib_cost(qty, price, False)) / price, 0.0)
    return qty


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


_GEO_RE = re.compile(
    r"\b("
    r"war|strike|strikes|military action|conflict|ceasefire|gulf|iran|israel|"
    r"oil supply|supply disruption|energy disruption|geopolitical|hormuz|"
    r"missile|attack|attacks|invasion|combat"
    r")\b",
    re.IGNORECASE,
)
_MACRO_RE = re.compile(
    r"\b("
    r"fed|federal reserve|rate|rates|cpi|inflation|recession|jobs|payroll|"
    r"policy|tariff|commodity|commodities|crude|oil|gas|energy|gold|dollar|"
    r"treasury|yield|yields"
    r")\b",
    re.IGNORECASE,
)


def event_family_from_text(question: Any, archetype: Any) -> str:
    """Classify a candidate using only text known at candidate time."""
    text = f"{archetype or ''} {question or ''}".lower()
    if _GEO_RE.search(text):
        return "geo"
    if "earnings" in text:
        return "earnings"
    if _MACRO_RE.search(text):
        return "macro"
    return "other"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _allocation_rank_tuple(trade: dict) -> tuple:
    """Live-safe rank: event family, entry probability, clipped runup, stable order."""
    priority = int(trade.get("event_priority", EVENT_PRIORITY_ORDER["other"]))
    entry_prob = _safe_float(trade.get("entry_prob"), 0.0)
    runup = _safe_float(trade.get("feat_runup_since_t0"), 0.0)
    clipped_runup = float(np.clip(runup, RANK_RUNUP_CLIP[0], RANK_RUNUP_CLIP[1]))
    return (
        priority,
        -entry_prob,
        -clipped_runup,
        int(trade.get("_candidate_order", 0)),
    )


def _annotate_allocation_fields(trade: dict, row: pd.Series, candidate_order: int) -> None:
    family = event_family_from_text(
        trade.get("question", row.get("question", "")),
        trade.get("archetype", row.get("feat_archetype", "")),
    )
    runup = _safe_float(row.get("feat_runup_since_t0"), 0.0)
    trade["_candidate_order"] = candidate_order
    trade["allocation_mode"] = ALLOCATION_EVENT_PRIORITY
    trade["event_family"] = family
    trade["event_priority"] = EVENT_PRIORITY_ORDER[family]
    trade["feat_runup_since_t0"] = runup
    trade["rank_runup_clipped"] = round(float(np.clip(runup, RANK_RUNUP_CLIP[0], RANK_RUNUP_CLIP[1])), 6)
    trade["entry_prob_rank_value"] = round(_safe_float(trade.get("entry_prob"), 0.0), 6)
    trade["supporting_market_ids"] = ""
    trade["supporting_questions"] = ""
    trade["same_day_rank"] = 0
    trade["entry_prob_rank"] = 0
    trade["runup_rank"] = 0
    trade["skip_reason"] = ""
    trade["preempted_trade_id"] = ""
    trade["preempt_reason"] = ""
    trade["duplicate_signal_upgrade"] = 0


def _diagnose_candidate_rejection(
    row: pd.Series, 
    policy: dict, 
    prices: dict, 
    probs: dict,
    candidate_order: int
) -> dict:
    sym, mkt = row["symbol"], row["market_id"]
    t_theta = as_utc_day(row["t_theta"])
    t_e = as_utc_day(row["t_e"])
    
    diag = {
        "candidate_t_theta": str(t_theta.date()),
        "entry_date_if_any": "",
        "market_id": mkt,
        "symbol": sym,
        "question": str(row.get("question", "")),
        "feat_runup_since_t0": _safe_float(row.get("feat_runup_since_t0"), 0.0),
        "entry_prob": "",
        "same_day_rank": "",
        "entry_prob_rank": "",
        "runup_rank": "",
        "disposition": "unknown",
        "preempted_trade_id": "",
        "preempt_reason": "",
        "_candidate_order": candidate_order,
    }

    # Mirror simulate_one exactly: a bearish-YES question is re-polarized (never
    # shorted), and a no-clean-side pair is skipped. Diverging here would make
    # the disposition log lie about why a candidate was rejected.
    question = str(row.get("question", ""))
    polarity, polarity_source = resolve_polarity(question, sym)
    diag["polarity"] = polarity
    diag["polarity_source"] = polarity_source
    if polarity == 0:
        diag["disposition"] = "no_clean_signal_side"
        return diag

    closes = prices.get(sym, [])
    win = [(t, h, l, c) for t, h, l, c in closes if t_theta - pd.Timedelta(days=30) <= t <= t_e]
    if len(win) < 2:
        diag["disposition"] = "insufficient_price_window"
        return diag

    mkt_probs = effective_prob_path(probs.get(mkt, []), polarity)
    if not mkt_probs:
        diag["disposition"] = "no_probability_data"
        return diag

    ent = entry_day(mkt_probs, t_theta, policy)
    if ent is None:
        diag["disposition"] = "below_entry_floor"
        return diag

    diag["entry_prob"] = round(ent[1], 3)

    p_surge = effective_prob_surge(row, polarity)
    if p_surge is not None and p_surge > policy.get("max_prob_surge", 999.0):
        diag["disposition"] = "prob_surge_exceeded"
        return diag
        
    r_surge = row.get("feat_runup_since_t0")
    if r_surge is not None and r_surge > policy.get("max_price_runup", 999.0):
        diag["disposition"] = "price_runup_exceeded"
        return diag
        
    entry_ts = ent[0]
    entry_idx = next((i for i, b in enumerate(win) if b[0] >= entry_ts), -1)
    if entry_idx == -1:
        diag["disposition"] = "entry_bar_unavailable"
        return diag
    path = win[entry_idx:]
    if len(path) < 2:
        diag["disposition"] = "entry_bar_unavailable"
        return diag
        
    entry_price = path[0][3]
    if entry_price == 0:
        diag["disposition"] = "zero_atr_or_price"
        return diag
        
    diag["disposition"] = "unknown_sim_rejection"
    return diag


def _make_disposition_row(
    trade: dict,
    policy: dict,
    open_positions_count: int | str,
    allocation_mode: str
) -> dict:
    family = trade.get("event_family", "")
    if not family and "question" in trade:
        family = event_family_from_text(trade.get("question", ""), trade.get("archetype", ""))
        
    row = {
        "candidate_t_theta": trade.get("candidate_t_theta", ""),
        "entry_date_if_any": trade.get("entry_date", trade.get("entry_date_if_any", "")),
        "market_id": trade.get("market_id", ""),
        "symbol": trade.get("symbol", ""),
        "question": trade.get("question", ""),
        "polarity": trade.get("polarity", ""),
        "polarity_source": trade.get("polarity_source", ""),
        "event_family": family,
        "event_priority": EVENT_PRIORITY_ORDER.get(family, EVENT_PRIORITY_ORDER["other"]) if allocation_mode == ALLOCATION_EVENT_PRIORITY else "",
        "entry_prob": trade.get("entry_prob", ""),
        "feat_runup_since_t0": trade.get("feat_runup_since_t0", ""),
        "same_day_rank": trade.get("same_day_rank", ""),
        "entry_prob_rank": trade.get("entry_prob_rank", ""),
        "runup_rank": trade.get("runup_rank", ""),
        "disposition": trade.get("skip_reason") or trade.get("disposition", "selected"),
        "enter_floor_at_candidate_time": policy.get("enter_floor", ""),
        "enter_strong_at_candidate_time": policy.get("enter_strong", ""),
        "open_positions_at_candidate_time": open_positions_count,
        "max_concurrent_at_candidate_time": policy.get("max_concurrent", ""),
        "preempted_trade_id": trade.get("preempted_trade_id", ""),
        "preempt_reason": trade.get("preempt_reason", ""),
    }
    return row


def _prepare_event_priority_batch(day_trades: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rank same-day candidates and collapse duplicate symbols into supporting signals."""
    if not day_trades:
        return [], []

    ranked = sorted(day_trades, key=_allocation_rank_tuple)
    day_frame = pd.DataFrame(
        {
            "_idx": list(range(len(ranked))),
            "entry_prob": [_safe_float(t.get("entry_prob"), 0.0) for t in ranked],
            "runup": [_safe_float(t.get("feat_runup_since_t0"), 0.0) for t in ranked],
        }
    )
    entry_prob_ranks = day_frame["entry_prob"].rank(method="min", ascending=False).astype(int).tolist()
    runup_ranks = day_frame["runup"].rank(method="min", ascending=False).astype(int).tolist()

    for idx, trade in enumerate(ranked):
        trade["same_day_rank"] = idx + 1
        trade["entry_prob_rank"] = entry_prob_ranks[idx]
        trade["runup_rank"] = runup_ranks[idx]

    winners: list[dict] = []
    collapsed: list[dict] = []
    seen_symbols: set[str] = set()
    for trade in ranked:
        symbol = str(trade.get("symbol", "")).upper()
        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            winners.append(trade)
            continue

        winner = next(t for t in winners if str(t.get("symbol", "")).upper() == symbol)
        support_ids = [x for x in str(winner.get("supporting_market_ids", "")).split("|") if x]
        support_questions = [x for x in str(winner.get("supporting_questions", "")).split(" || ") if x]
        support_ids.append(str(trade.get("market_id", "")))
        support_questions.append(str(trade.get("question", "")))
        winner["supporting_market_ids"] = "|".join(support_ids)
        winner["supporting_questions"] = " || ".join(support_questions)
        trade["skip_reason"] = "same_day_symbol_collapsed"
        collapsed.append(trade)

    return winners, collapsed


# ── Policy / Kelly helpers ───────────────────────────────────────────────────

def port_policy_from_vec(vec: np.ndarray) -> dict[str, float | int]:
    names = list(PORTFOLIO_BOUNDS.keys())
    policy: dict[str, float | int] = {}
    for i, name in enumerate(names):
        lo, hi = PORTFOLIO_BOUNDS[name]
        policy[name] = float(np.clip(vec[i], lo, hi))

    policy["hold_days"] = int(round(float(policy["hold_days"])))
    policy["max_concurrent"] = int(round(float(policy["max_concurrent"])))
    if float(policy["enter_strong"]) < float(policy["enter_floor"]):
        policy["enter_strong"] = policy["enter_floor"]
    return policy


def kelly_size(completed_history: list[dict], base: float) -> float:
    """Half-Kelly from the latest fully net realised trades."""
    if len(completed_history) < KELLY_MIN_N:
        return base

    recent = completed_history[-KELLY_LOOKBACK_N:]
    wins = [float(t["pnl_pct"]) for t in recent if float(t.get("pnl_pct", 0.0)) > 0]
    losses = [float(t["pnl_pct"]) for t in recent if float(t.get("pnl_pct", 0.0)) <= 0]

    if not wins or not losses:
        return base

    win_probability = len(wins) / len(recent)
    payoff_ratio = abs(float(np.mean(wins))) / abs(float(np.mean(losses)))
    if payoff_ratio <= 0 or not np.isfinite(payoff_ratio):
        return base

    full_kelly = (win_probability * payoff_ratio - (1.0 - win_probability)) / payoff_ratio
    half_kelly = max(0.0, full_kelly / 2.0)
    return float(np.clip(half_kelly, KELLY_MIN_SZ, KELLY_MAX_SZ))


# ── Expanding walk-forward folds ─────────────────────────────────────────────

def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    """Convert a timestamp-like value to UTC without discarding its time."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _frame_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.to_datetime(df["t_theta"], utc=True)
    return as_utc_day(ts.min()), as_utc_day(ts.max())


def _calc_advanced_metrics(equity_series: pd.Series) -> dict[str, float]:
    if len(equity_series) < 2:
        return {"sharpe": 0.0, "sortino": 0.0}
    r = equity_series.pct_change().dropna()
    mu = r.mean()
    sig = r.std()
    sharpe = (mu / sig * np.sqrt(252)) if sig > 1e-9 else 0.0
    down = r[r < 0]
    down_sig = down.std() if len(down) > 0 else 1e-9
    sortino = (mu / down_sig * np.sqrt(252)) if down_sig > 1e-9 else 0.0
    return {"sharpe": float(sharpe), "sortino": float(sortino)}


def _calc_max_dd(equity_series: pd.Series) -> float:
    vals = equity_series.astype(float).to_numpy()
    if len(vals) == 0:
        return 0.0
    peaks = np.maximum.accumulate(vals)
    drawdowns = np.where(peaks > 0, vals / peaks - 1.0, 0.0)
    return float(np.min(drawdowns) * 100.0)


def _print_monthly_metrics(equity_df: pd.DataFrame, label: str, benchmark: str) -> None:
    if equity_df.empty:
        return
    
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")
    
    months = df["year_month"].unique()
    
    print("\n    [Monthly Progress]", flush=True)
    for m in months:
        m_df = df[df["year_month"] == m]
        c_df = df[df["year_month"] <= m]
        
        m_strat_ret = (m_df["equity"].iloc[-1] / m_df["equity"].iloc[0] - 1.0) * 100.0 if not m_df.empty and m_df["equity"].iloc[0] > 0 else 0.0
        m_strat_adv = _calc_advanced_metrics(m_df["equity"])
        m_strat_dd = _calc_max_dd(m_df["equity"])
        
        m_bench_ret = (m_df["benchmark_equity"].iloc[-1] / m_df["benchmark_equity"].iloc[0] - 1.0) * 100.0 if not m_df.empty and m_df["benchmark_equity"].iloc[0] > 0 else 0.0
        m_bench_adv = _calc_advanced_metrics(m_df["benchmark_equity"])
        m_bench_dd = _calc_max_dd(m_df["benchmark_equity"])
        
        c_strat_ret = (c_df["equity"].iloc[-1] / c_df["equity"].iloc[0] - 1.0) * 100.0 if not c_df.empty and c_df["equity"].iloc[0] > 0 else 0.0
        c_strat_adv = _calc_advanced_metrics(c_df["equity"])
        c_strat_dd = _calc_max_dd(c_df["equity"])
        
        c_bench_ret = (c_df["benchmark_equity"].iloc[-1] / c_df["benchmark_equity"].iloc[0] - 1.0) * 100.0 if not c_df.empty and c_df["benchmark_equity"].iloc[0] > 0 else 0.0
        c_bench_adv = _calc_advanced_metrics(c_df["benchmark_equity"])
        c_bench_dd = _calc_max_dd(c_df["benchmark_equity"])

        month_str = m.strftime("%B %Y")
        
        print(f"      In {month_str}: {label} [Ret: {m_strat_ret:+.2f}%, DD: {m_strat_dd:+.2f}%, Shp: {m_strat_adv['sharpe']:.2f}]  |  "
              f"{benchmark} B&H [Ret: {m_bench_ret:+.2f}%, DD: {m_bench_dd:+.2f}%, Shp: {m_bench_adv['sharpe']:.2f}]", flush=True)
        print(f"      Total Start->{m.strftime('%b')}: {label} [Ret: {c_strat_ret:+.2f}%, DD: {c_strat_dd:+.2f}%, Shp: {c_strat_adv['sharpe']:.2f}]  |  "
              f"{benchmark} B&H [Ret: {c_bench_ret:+.2f}%, DD: {c_bench_dd:+.2f}%, Shp: {c_bench_adv['sharpe']:.2f}]\n", flush=True)


def rows_completed_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Return the rows legally available for fitting at `cutoff`.

    This is deliberately based on t_e (label/outcome completion), not t_theta
    (candidate/entry time). The comparison is strict: completion on the fold
    start is not "before" the fold start.
    """
    cutoff_ts = _as_utc_timestamp(cutoff)
    completed_at = pd.to_datetime(df["t_e"], utc=True)
    return df.loc[completed_at < cutoff_ts].copy()


def assert_rows_completed_before(
    df: pd.DataFrame,
    cutoff: Any,
    *,
    context: str,
) -> None:
    """Fail closed if a CEM fit contains any label incomplete at its cutoff."""
    if df.empty:
        raise ValueError(f"{context}: no rows are available for fitting.")

    cutoff_ts = _as_utc_timestamp(cutoff)
    completed_at = pd.to_datetime(df["t_e"], utc=True)
    invalid = completed_at >= cutoff_ts
    if invalid.any():
        first_bad = completed_at.loc[invalid].min()
        raise ValueError(
            f"{context}: {int(invalid.sum())} fit row(s) have t_e >= "
            f"{cutoff_ts.isoformat()}; earliest invalid t_e={first_bad.isoformat()}. "
            "Training eligibility must be determined by t_e < fold start."
        )


def create_expanding_wf_folds(
    train_df: pd.DataFrame,
    final_fit_cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    """
    Construct expanding, label-complete walk-forward folds from the train split.

    Fold i:
      fit rows  = every train-split candidate with t_e < fold_i.eval_start
      eval rows = candidates with t_theta in [eval_start, eval_end_exclusive)

    `t_e`, not `t_theta`, controls fit eligibility. Evaluation is still blocked
    by `t_theta` because it represents the time at which a candidate becomes
    tradeable. The fold's portfolio simulation is truncated at eval_end, so
    neither fit nor evaluation reads data beyond its own decision horizon.
    """
    if train_df.empty:
        return []

    theta = pd.to_datetime(train_df["t_theta"], utc=True)
    t_e = pd.to_datetime(train_df["t_e"], utc=True)
    final_cutoff = as_utc_day(final_fit_cutoff)

    first_candidate_day = as_utc_day(theta.min())
    # The split itself is already chronological, but use the earliest of its
    # final fit cutoff and last candidate day + 1 to make the end exclusive.
    last_eval_exclusive = min(
        final_cutoff,
        as_utc_day(theta.max()) + pd.Timedelta(days=1),
    )

    folds: list[dict[str, Any]] = []
    eval_start = first_candidate_day

    while eval_start < last_eval_exclusive:
        eval_end_exclusive = min(
            as_utc_day(eval_start + pd.DateOffset(months=WF_EVAL_MON)),
            last_eval_exclusive,
        )
        if eval_end_exclusive <= eval_start:
            break

        fit_df = train_df.loc[t_e < eval_start].copy()
        eval_df = train_df.loc[
            (theta >= eval_start) & (theta < eval_end_exclusive)
        ].copy()

        if len(fit_df) >= WF_MIN_TRAIN_CANDS and len(eval_df) >= WF_MIN_EVAL_CANDS:
            assert_rows_completed_before(
                fit_df,
                eval_start,
                context=f"walk-forward fold {len(folds) + 1}",
            )
            fit_start, _ = _frame_bounds(fit_df)
            folds.append(
                {
                    "fold": len(folds) + 1,
                    "fit_df": fit_df,
                    "fit_start": fit_start,
                    "fit_cutoff": eval_start,
                    "fit_end": eval_start - pd.Timedelta(days=1),
                    "eval_df": eval_df,
                    "eval_start": eval_start,
                    "eval_end": eval_end_exclusive - pd.Timedelta(days=1),
                    "eval_end_exclusive": eval_end_exclusive,
                }
            )

        eval_start = as_utc_day(eval_start + pd.DateOffset(months=WF_STEP_MON))

    return folds


# ── Core portfolio simulator ─────────────────────────────────────────────────

def sim_opp_cost(
    df: pd.DataFrame,
    prices: dict,
    probs: dict,
    policy: dict | Callable[[pd.Timestamp], dict],
    *,
    bench_sym: str = "SPY",
    initial: float = INITIAL_CAPITAL,
    use_kelly: bool = False,
    start_date: Any | None = None,
    end_date: Any | None = None,
    initial_kelly_history: list[dict] | None = None,
    allocation_mode: str = ALLOCATION_FIFO,
    collect_allocation_log: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict, pd.DataFrame, pd.DataFrame]:
    """
    Simulate a benchmark-rotation portfolio.

    When end_date is supplied, open positions are liquidated at that date's
    available daily close. This is essential for train windows: no score can
    depend on price/probability data after its own evaluation horizon.

    The returned per-trade `pnl` is fully net of all modeled costs associated
    with rotating benchmark capital into and out of the asset.
    """
    if allocation_mode not in {ALLOCATION_FIFO, ALLOCATION_EVENT_PRIORITY}:
        raise ValueError(f"Unsupported allocation_mode={allocation_mode!r}.")

    base_ps_static = float(policy.get("position_size_pct", 0.10)) if isinstance(policy, dict) else 0.10
    max_concurrent_static = int(policy.get("max_concurrent", 10)) if isinstance(policy, dict) else 10

    empty_stats: dict[str, Any] = {
        "initial": initial,
        "final": initial,
        "total_return": 0.0,
        "benchmark_return": 0.0,
        "excess_return": 0.0,
        "max_dd": 0.0,
        "n_trades": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "avg_gross_pnl": 0.0,
        "gross_trade_pnl": 0.0,
        "net_trade_pnl": 0.0,
        "total_txn_cost": 0.0,
        "trade_txn_cost": 0.0,
        "friction_fail_rate": 0.0,
        "avg_position_size": 0.0,
        "median_position_size": 0.0,
        "min_position_size": 0.0,
        "max_position_size": 0.0,
        "start_date": None,
        "end_date": None,
        "n_equity_days": 0,
        "skip_max_concurrent": 0,
        "skip_duplicate_symbol": 0,
        "skip_insufficient_capital": 0,
        "skip_same_day_symbol_collapsed": 0,
        "skip_preempt_hurdle": 0,
        "preemptions": 0,
        "selected_geo_exposure_share": 0.0,
        "selected_macro_exposure_share": 0.0,
        "selected_earnings_exposure_share": 0.0,
        "selected_other_exposure_share": 0.0,
        "selected_geo_trades": 0,
        "selected_macro_trades": 0,
        "selected_earnings_trades": 0,
        "selected_other_trades": 0,
        "skipped_high_priority_opportunities": 0,
        "skipped_geo_opportunities": 0,
        "skipped_macro_opportunities": 0,
        "allocation_mode": allocation_mode,
    }
    if df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            empty_stats,
            (policy if isinstance(policy, dict) else {}),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    sim_prices, sim_probs = truncate_paths(prices, probs, end_date)

    all_trades: list[dict] = []
    candidate_disposition_rows: list[dict[str, Any]] = []
    for candidate_order, (_, row) in enumerate(df.sort_values("t_theta").iterrows(), start=1):
        candidate_theta = as_utc_day(row["t_theta"])
        current_policy = policy(candidate_theta) if callable(policy) else policy
        trade = simulate_one(row, sim_prices, sim_probs, current_policy)
        if trade is None:
            if collect_allocation_log:
                diag = _diagnose_candidate_rejection(row, current_policy, sim_prices, sim_probs, candidate_order)
                candidate_disposition_rows.append(_make_disposition_row(diag, current_policy, "", allocation_mode))
            continue
        trade = dict(trade)
        trade["_entry_ts"] = as_utc_day(trade["entry_date"])
        trade["_exit_ts"] = as_utc_day(trade["exit_date"])
        if trade["_entry_ts"] < candidate_theta:
            raise ValueError(
                f"{trade['symbol']} entered on {trade['_entry_ts'].date()} before "
                f"candidate t_theta {candidate_theta.date()}."
            )
        trade["candidate_t_theta"] = str(candidate_theta.date())
        trade["candidate_t_e"] = str(as_utc_day(row["t_e"]).date())
        if allocation_mode == ALLOCATION_EVENT_PRIORITY:
            _annotate_allocation_fields(trade, row, candidate_order)
        all_trades.append(trade)
    all_trades.sort(key=lambda trade: trade["_entry_ts"])

    candidate_start, candidate_end = _frame_bounds(df)
    eval_start = as_utc_day(start_date) if start_date is not None else (
        min((trade["_entry_ts"] for trade in all_trades), default=candidate_start)
    )
    if end_date is not None:
        eval_end = as_utc_day(end_date)
    else:
        eval_end = max((trade["_exit_ts"] for trade in all_trades), default=candidate_end)

    if end_date is not None:
        future_exits = [trade for trade in all_trades if trade["_exit_ts"] > eval_end]
        if future_exits:
            raise ValueError(
                f"{len(future_exits)} generated trades exit after evaluation end "
                f"{eval_end.date()}."
            )

    bench_bars = sim_prices.get(bench_sym, [])
    if not bench_bars:
        raise ValueError(f"No daily bars available for benchmark {bench_sym}.")

    calendar = _calendar_dates(sim_prices, bench_sym, eval_start, eval_end)
    if not calendar:
        raise ValueError(
            f"No {bench_sym} daily bars overlap requested evaluation range "
            f"{eval_start.date()} through {eval_end.date()}."
        )

    first_day = calendar[0]
    last_day = calendar[-1]
    first_bench_close = _close_on(sim_prices, bench_sym, first_day)
    last_bench_close = _close_on(sim_prices, bench_sym, last_day)
    if first_bench_close is None or last_bench_close is None:
        raise ValueError(f"Unable to price {bench_sym} across evaluation dates.")

    # Establish the portfolio and the passive benchmark using exactly the same
    # initial execution model.
    initial_bench_shares = _bench_buy_qty(initial, first_bench_close)
    initial_cost = ib_cost(initial_bench_shares, first_bench_close, False)
    initial_cash = initial - initial_bench_shares * first_bench_close - initial_cost

    bench_shares = initial_bench_shares
    cash = initial_cash
    total_txn_cost = initial_cost

    open_positions: list[dict] = []
    completed: list[dict] = []
    kelly_history = [dict(item) for item in (initial_kelly_history or [])]
    equity_rows: list[dict[str, Any]] = []
    trade_idx = 0
    skip_max_concurrent = 0
    skip_duplicate_symbol = 0
    skip_insufficient_capital = 0
    skip_same_day_symbol_collapsed = 0
    skip_preempt_hurdle = 0
    preemptions = 0
    allocation_rows: list[dict[str, Any]] = []
    next_allocation_trade_id = 1

    def close_position(pos: dict, close_day: pd.Timestamp, exit_price: float, exit_reason: str) -> None:
        """Close an asset position, rotate proceeds back to benchmark, and record net P&L."""
        nonlocal cash, bench_shares, total_txn_cost

        qty = int(pos["_qty"])
        entry_price = float(pos["entry_price"])
        asset_sell_cost = ib_cost(qty, exit_price, True)
        sale_proceeds = qty * exit_price - asset_sell_cost

        rebuy_qty = _bench_buy_qty(sale_proceeds, float(_close_on(sim_prices, bench_sym, close_day) or 0.0))
        bench_close = float(_close_on(sim_prices, bench_sym, close_day) or 0.0)
        rebuy_cost = ib_cost(rebuy_qty, bench_close, False)

        cash += sale_proceeds - rebuy_qty * bench_close - rebuy_cost
        bench_shares += rebuy_qty
        total_txn_cost += asset_sell_cost + rebuy_cost

        direct_cost = (
            float(pos["_benchmark_sell_cost"])
            + float(pos["_asset_buy_cost"])
            + asset_sell_cost
            + rebuy_cost
        )
        gross_pnl = qty * (exit_price - entry_price)
        net_pnl = gross_pnl - direct_cost
        exposure = max(float(pos["_asset_entry_notional"]), 1e-12)

        pos["exit_price"] = round(float(exit_price), 6)
        pos["exit_date"] = str(close_day.date())
        pos["realized_exit_reason"] = exit_reason
        pos["gross_pnl"] = round(gross_pnl, 2)
        pos["pnl"] = round(net_pnl, 2)
        pos["pnl_pct"] = round(net_pnl / exposure * 100.0, 4)
        pos["txn_cost"] = round(direct_cost, 2)
        pos["exit_value"] = round(qty * exit_price, 2)
        pos["benchmark_rebuy_qty"] = round(rebuy_qty, 4)

        completed.append(pos)
        kelly_history.append(pos)

    def sweep_idle_cash(bench_close: float) -> None:
        """Redeploy idle cash into benchmark shares (fully-invested rule)."""
        nonlocal cash, bench_shares, total_txn_cost
        if not FULLY_INVESTED_SWEEP or bench_close <= 0 or cash < MIN_SWEEP_CASH:
            return
        qty = _bench_buy_qty(cash, bench_close)
        if qty <= 0:
            return
        buy_cost = ib_cost(qty, bench_close, False)
        cash -= qty * bench_close + buy_cost
        bench_shares += qty
        total_txn_cost += buy_cost

    def record_allocation_decision(
        trade: dict,
        day: pd.Timestamp,
        *,
        decision: str,
        skip_reason: str = "",
        preempted_trade_id: str = "",
        preempt_reason: str = "",
        open_before: int | None = None,
        open_after: int | None = None,
        current_equity: float | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        if not collect_allocation_log:
            return
        allocation_rows.append(
            {
                "date": str(day.date()),
                "allocation_mode": allocation_mode,
                "decision": decision,
                "skip_reason": skip_reason,
                "market_id": trade.get("market_id", ""),
                "symbol": trade.get("symbol", ""),
                "question": trade.get("question", ""),
                "event_family": trade.get("event_family", ""),
                "event_priority": trade.get("event_priority", ""),
                "same_day_rank": trade.get("same_day_rank", ""),
                "entry_prob_rank": trade.get("entry_prob_rank", ""),
                "runup_rank": trade.get("runup_rank", ""),
                "entry_prob": trade.get("entry_prob", ""),
                "feat_runup_since_t0": trade.get("feat_runup_since_t0", ""),
                "rank_runup_clipped": trade.get("rank_runup_clipped", ""),
                "entry_date": trade.get("entry_date", ""),
                "exit_date": trade.get("exit_date", ""),
                "candidate_t_theta": trade.get("candidate_t_theta", ""),
                "candidate_t_e": trade.get("candidate_t_e", ""),
                "exit_reason": trade.get("exit_reason", ""),
                "supporting_market_ids": trade.get("supporting_market_ids", ""),
                "supporting_questions": trade.get("supporting_questions", ""),
                "allocation_trade_id": trade.get("allocation_trade_id", ""),
                "preempted_trade_id": preempted_trade_id or trade.get("preempted_trade_id", ""),
                "preempt_reason": preempt_reason or trade.get("preempt_reason", ""),
                "open_positions_before": open_before,
                "open_positions_after": open_after,
                "max_concurrent": max_concurrent if max_concurrent is not None else max_concurrent_static,
                "current_equity": round(current_equity, 2) if current_equity is not None else "",
            }
        )
        pol = policy(as_utc_day(trade.get("candidate_t_theta", day))) if callable(policy) else policy
        candidate_disposition_rows.append(_make_disposition_row(trade, pol, open_before if open_before is not None else "", allocation_mode))

    def current_position_net_pct(pos: dict, day: pd.Timestamp, bench_close: float) -> tuple[float, float]:
        qty = int(pos["_qty"])
        mark_price = float(_close_on(sim_prices, pos["symbol"], day) or pos["entry_price"])
        asset_sell_cost = ib_cost(qty, mark_price, True)
        sale_proceeds = qty * mark_price - asset_sell_cost
        rebuy_qty = _bench_buy_qty(sale_proceeds, bench_close)
        rebuy_cost = ib_cost(rebuy_qty, bench_close, False)
        direct_cost = (
            float(pos.get("_benchmark_sell_cost", 0.0))
            + float(pos.get("_asset_buy_cost", 0.0))
            + asset_sell_cost
            + rebuy_cost
        )
        gross_pnl = qty * (mark_price - float(pos["entry_price"]))
        exposure = max(float(pos.get("_asset_entry_notional", 0.0)), 1e-12)
        return (gross_pnl - direct_cost) / exposure * 100.0, mark_price

    def maybe_preempt_earnings_slot(
        trade: dict,
        day: pd.Timestamp,
        bench_close: float,
    ) -> str | None:
        nonlocal open_positions, skip_preempt_hurdle, preemptions

        if allocation_mode != ALLOCATION_EVENT_PRIORITY:
            return None
        if trade.get("event_family") not in {"geo", "macro"}:
            return None
        if not open_positions or not all(pos.get("event_family") == "earnings" for pos in open_positions):
            return None

        scored: list[tuple[float, float, dict]] = []
        for pos in open_positions:
            net_pct, mark_price = current_position_net_pct(pos, day, bench_close)
            scored.append((net_pct, mark_price, pos))
        if not scored:
            return None

        worst_net_pct, worst_price, worst_pos = min(scored, key=lambda item: item[0])
        if worst_net_pct >= PREEMPT_NET_PROFIT_HURDLE_PCT:
            skip_preempt_hurdle += 1
            return None

        preempted_trade_id = str(worst_pos.get("allocation_trade_id", ""))
        reason = (
            f"preempted_by_{trade.get('event_family')}_under_"
            f"{PREEMPT_NET_PROFIT_HURDLE_PCT:.0f}pct_net_hurdle"
        )
        worst_pos["preempted_by_market_id"] = trade.get("market_id", "")
        worst_pos["preempted_by_symbol"] = trade.get("symbol", "")
        worst_pos["preempted_by_event_family"] = trade.get("event_family", "")
        worst_pos["preempt_reason"] = reason
        close_position(
            worst_pos,
            close_day=day,
            exit_price=float(worst_price),
            exit_reason="preempted_by_event_priority",
        )
        open_positions = [pos for pos in open_positions if pos is not worst_pos]
        preemptions += 1
        return preempted_trade_id

    def try_open_trade(
        trade: dict,
        day: pd.Timestamp,
        bench_close: float,
        *,
        base_ps: float,
        max_concurrent: int,
    ) -> bool:
        nonlocal cash, bench_shares, total_txn_cost, next_allocation_trade_id
        nonlocal skip_max_concurrent, skip_duplicate_symbol, skip_insufficient_capital

        open_before = len(open_positions)
        preempted_trade_id = ""
        preempt_reason = ""

        if len(open_positions) >= max_concurrent:
            preempted_trade_id = maybe_preempt_earnings_slot(trade, day, bench_close) or ""
            if preempted_trade_id:
                preempt_reason = (
                    f"preempted_earnings_below_{PREEMPT_NET_PROFIT_HURDLE_PCT:.0f}pct_net_hurdle"
                )
            else:
                skip_max_concurrent += 1
                trade["skip_reason"] = "max_concurrent"
                record_allocation_decision(
                    trade,
                    day,
                    decision="skipped",
                    skip_reason="max_concurrent",
                    open_before=open_before,
                    open_after=len(open_positions),
                    max_concurrent=max_concurrent,
                )
                return False

        if any(pos["symbol"] == trade["symbol"] for pos in open_positions):
            skip_duplicate_symbol += 1
            trade["skip_reason"] = "duplicate_symbol"
            record_allocation_decision(
                trade,
                day,
                decision="skipped",
                skip_reason="duplicate_symbol",
                open_before=open_before,
                open_after=len(open_positions),
                max_concurrent=max_concurrent,
            )
            return False

        position_size = kelly_size(kelly_history, base_ps) if use_kelly else base_ps
        marked_open_value = sum(
            int(pos["_qty"]) * float(_close_on(sim_prices, pos["symbol"], day) or pos["entry_price"])
            for pos in open_positions
        )
        current_equity = bench_shares * bench_close + marked_open_value + cash
        desired_allocation = current_equity * position_size
        entry_price = float(trade["entry_price"])
        if entry_price <= 0 or desired_allocation < entry_price:
            skip_insufficient_capital += 1
            trade["skip_reason"] = "insufficient_capital"
            record_allocation_decision(
                trade,
                day,
                decision="skipped",
                skip_reason="insufficient_capital",
                open_before=open_before,
                open_after=len(open_positions),
                current_equity=current_equity,
                max_concurrent=max_concurrent,
            )
            return False

        # Fund the entry from idle cash first, then benchmark inventory.
        cash_contribution = min(max(cash, 0.0), desired_allocation)
        shortfall = desired_allocation - cash_contribution
        if shortfall > 0:
            desired_sell = (
                shortfall / bench_close if FRACTIONAL_BENCHMARK
                else int(shortfall / bench_close)
            )
            benchmark_sell_qty = min(desired_sell, bench_shares)
        else:
            benchmark_sell_qty = 0.0
        if cash_contribution + benchmark_sell_qty * bench_close < entry_price:
            skip_insufficient_capital += 1
            trade["skip_reason"] = "insufficient_capital"
            record_allocation_decision(
                trade,
                day,
                decision="skipped",
                skip_reason="insufficient_capital",
                open_before=open_before,
                open_after=len(open_positions),
                current_equity=current_equity,
                max_concurrent=max_concurrent,
            )
            return False

        benchmark_sell_cost = (
            ib_cost(benchmark_sell_qty, bench_close, True) if benchmark_sell_qty > 0 else 0.0
        )
        available_for_asset = (
            cash_contribution + benchmark_sell_qty * bench_close - benchmark_sell_cost
        )
        asset_qty = _affordable_buy_qty(available_for_asset, entry_price)
        if asset_qty < 1:
            skip_insufficient_capital += 1
            trade["skip_reason"] = "insufficient_capital"
            record_allocation_decision(
                trade,
                day,
                decision="skipped",
                skip_reason="insufficient_capital",
                open_before=open_before,
                open_after=len(open_positions),
                current_equity=current_equity,
                max_concurrent=max_concurrent,
            )
            return False

        asset_buy_cost = ib_cost(asset_qty, entry_price, False)
        asset_cash_needed = asset_qty * entry_price + asset_buy_cost
        if asset_cash_needed > available_for_asset + 1e-9:
            skip_insufficient_capital += 1
            trade["skip_reason"] = "insufficient_capital"
            record_allocation_decision(
                trade,
                day,
                decision="skipped",
                skip_reason="insufficient_capital",
                open_before=open_before,
                open_after=len(open_positions),
                current_equity=current_equity,
                max_concurrent=max_concurrent,
            )
            return False

        bench_shares -= benchmark_sell_qty
        cash += available_for_asset - asset_cash_needed - cash_contribution
        total_txn_cost += benchmark_sell_cost + asset_buy_cost

        opened_trade = {
            **trade,
            "_qty": asset_qty,
            "_position_size_pct": position_size,
            "_asset_entry_notional": round(asset_qty * entry_price, 2),
            "_equity_at_entry": round(current_equity, 2),
            "invested_frac_pct": round(
                asset_qty * entry_price / max(current_equity, 1e-12) * 100.0, 4
            ),
            "_benchmark_sell_cost": round(benchmark_sell_cost, 2),
            "_asset_buy_cost": round(asset_buy_cost, 2),
            "_entry_ts": trade["_entry_ts"],
            "_exit_ts": trade["_exit_ts"],
        }
        if allocation_mode == ALLOCATION_EVENT_PRIORITY:
            opened_trade["allocation_trade_id"] = next_allocation_trade_id
            opened_trade["preempted_trade_id"] = preempted_trade_id
            opened_trade["preempt_reason"] = preempt_reason
            next_allocation_trade_id += 1

        open_positions.append(opened_trade)
        record_allocation_decision(
            opened_trade,
            day,
            decision="selected",
            preempted_trade_id=preempted_trade_id,
            preempt_reason=preempt_reason,
            open_before=open_before,
            open_after=len(open_positions),
            current_equity=current_equity,
            max_concurrent=max_concurrent,
        )
        return True

    for day in calendar:
        current_policy = policy(day) if callable(policy) else policy
        base_ps = float(current_policy.get("position_size_pct", 0.10))
        max_concurrent = int(current_policy.get("max_concurrent", 10))

        bench_close = _close_on(sim_prices, bench_sym, day)
        if bench_close is None:
            continue

        # Close trades whose planned exit is known by the current day.
        still_open: list[dict] = []
        for pos in open_positions:
            if pos["_exit_ts"] <= day:
                exit_reason = str(pos.get("exit_reason", "strategy_exit"))
                if end_date is not None and exit_reason == "end_of_window":
                    exit_reason = "evaluation_end_liquidation"
                close_position(
                    pos,
                    close_day=day,
                    exit_price=float(pos["exit_price"]),
                    exit_reason=exit_reason,
                )
            else:
                still_open.append(pos)
        open_positions = still_open

        # Open candidates exactly on their configured entry day.
        if allocation_mode == ALLOCATION_FIFO:
            while trade_idx < len(all_trades):
                trade = all_trades[trade_idx]
                if trade["_entry_ts"] > day:
                    break
                trade_idx += 1

                if trade["_entry_ts"] < day:
                    gap_days = (day - trade["_entry_ts"]).days
                    trade["calendar_gap_days"] = gap_days
                    if gap_days > 4:
                        if collect_allocation_log:
                            trade["skip_reason"] = "excessive_calendar_gap"
                            record_allocation_decision(
                                trade, day, decision="skipped", skip_reason="excessive_calendar_gap",
                                open_before=len(open_positions), open_after=len(open_positions), max_concurrent=max_concurrent
                            )
                        continue
                    trade["_entry_ts"] = day
                    trade["entry_date"] = str(day.date())
                try_open_trade(
                    trade,
                    day,
                    float(bench_close),
                    base_ps=base_ps,
                    max_concurrent=max_concurrent,
                )
        else:
            daily_trades: list[dict] = []
            while trade_idx < len(all_trades):
                trade = all_trades[trade_idx]
                if trade["_entry_ts"] > day:
                    break
                trade_idx += 1

                if trade["_entry_ts"] < day:
                    gap_days = (day - trade["_entry_ts"]).days
                    trade["calendar_gap_days"] = gap_days
                    if gap_days > 4:
                        if collect_allocation_log:
                            trade["skip_reason"] = "excessive_calendar_gap"
                            record_allocation_decision(
                                trade, day, decision="skipped", skip_reason="excessive_calendar_gap",
                                open_before=len(open_positions), open_after=len(open_positions), max_concurrent=max_concurrent
                            )
                        continue
                    trade["_entry_ts"] = day
                    trade["entry_date"] = str(day.date())
                daily_trades.append(trade)

            ranked_trades, collapsed_trades = _prepare_event_priority_batch(daily_trades)
            for collapsed in collapsed_trades:
                skip_same_day_symbol_collapsed += 1
                record_allocation_decision(
                    collapsed,
                    day,
                    decision="skipped",
                    skip_reason="same_day_symbol_collapsed",
                    open_before=len(open_positions),
                    open_after=len(open_positions),
                    max_concurrent=max_concurrent,
                )
            for trade in ranked_trades:
                try_open_trade(
                    trade,
                    day,
                    float(bench_close),
                    base_ps=base_ps,
                    max_concurrent=max_concurrent,
                )

        sweep_idle_cash(float(bench_close))

        open_value = sum(
            int(pos["_qty"]) * float(_close_on(sim_prices, pos["symbol"], day) or pos["entry_price"])
            for pos in open_positions
        )
        equity = bench_shares * bench_close + open_value + cash
        passive_benchmark_equity = initial_bench_shares * bench_close + initial_cash
        equity_rows.append(
            {
                "date": str(day.date()),
                "equity": round(equity, 2),
                "benchmark_equity": round(passive_benchmark_equity, 2),
                "cash": round(cash, 2),
                "benchmark_shares": round(bench_shares, 4),
                "open_positions": len(open_positions),
            }
        )

    # If a finite evaluation horizon was supplied, mark positions to market and
    # liquidate on that horizon. This prevents CEM train windows from using
    # observations after their end date.
    if open_positions:
        for pos in list(open_positions):
            forced_price = _close_on(sim_prices, pos["symbol"], last_day)
            if forced_price is None:
                # This should not happen when the candidate universe and price
                # loader are consistent. Falling back to entry price is safer
                # than inventing a future close.
                forced_price = float(pos["entry_price"])
            close_position(
                pos,
                close_day=last_day,
                exit_price=float(forced_price),
                exit_reason="evaluation_end_liquidation",
            )
        open_positions = []

    final_equity = bench_shares * last_bench_close + cash
    final_passive_equity = initial_bench_shares * last_bench_close + initial_cash

    # The final forced liquidation costs occur after the final intraday mark.
    # Update the final row so drawdown and reported final equity agree.
    if equity_rows:
        equity_rows[-1]["equity"] = round(final_equity, 2)
        equity_rows[-1]["benchmark_equity"] = round(final_passive_equity, 2)
        equity_rows[-1]["cash"] = round(cash, 2)
        equity_rows[-1]["benchmark_shares"] = round(bench_shares, 4)
        equity_rows[-1]["open_positions"] = 0
    else:
        equity_rows.append(
            {
                "date": str(last_day.date()),
                "equity": round(final_equity, 2),
                "benchmark_equity": round(final_passive_equity, 2),
                "cash": round(cash, 2),
                "benchmark_shares": round(bench_shares, 4),
                "open_positions": 0,
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trade_df = pd.DataFrame(completed)

    equity_values = equity_df["equity"].astype(float).to_numpy()
    peaks = np.maximum.accumulate(equity_values)
    drawdowns = np.where(peaks > 0, equity_values / peaks - 1.0, 0.0)
    max_dd = float(np.min(drawdowns) * 100.0) if len(drawdowns) else 0.0

    if not equity_df.empty:
        adv = _calc_advanced_metrics(equity_df["equity"])
        bench_adv = _calc_advanced_metrics(equity_df["benchmark_equity"])
        strat_sharpe = adv["sharpe"]
        strat_sortino = adv["sortino"]
        bench_sharpe = bench_adv["sharpe"]
        bench_sortino = bench_adv["sortino"]
    else:
        strat_sharpe = strat_sortino = bench_sharpe = bench_sortino = 0.0

    if trade_df.empty:
        gross_trade_pnl = net_trade_pnl = trade_txn_cost = 0.0
        win_rate = avg_pnl = avg_gross_pnl = friction_fail_rate = 0.0
        position_sizes = np.asarray([], dtype=float)
        selected_family_stats = {
            "selected_geo_exposure_share": 0.0,
            "selected_macro_exposure_share": 0.0,
            "selected_earnings_exposure_share": 0.0,
            "selected_other_exposure_share": 0.0,
            "selected_geo_trades": 0,
            "selected_macro_trades": 0,
            "selected_earnings_trades": 0,
            "selected_other_trades": 0,
        }
    else:
        gross_trade_pnl = float(trade_df["gross_pnl"].sum())
        net_trade_pnl = float(trade_df["pnl"].sum())
        trade_txn_cost = float(trade_df["txn_cost"].sum())
        win_rate = float((trade_df["pnl"] > 0).mean() * 100.0)
        avg_pnl = float(trade_df["pnl"].mean())
        avg_gross_pnl = float(trade_df["gross_pnl"].mean())
        friction_fail_rate = float(
            (trade_df["gross_pnl"] < HURDLE_MULT * trade_df["txn_cost"]).mean() * 100.0
        )
        position_sizes = trade_df["_position_size_pct"].astype(float).to_numpy()
        family_series = trade_df.apply(
            lambda row: str(row.get("event_family", ""))
            or event_family_from_text(
                row.get("question", ""),
                row.get("archetype", row.get("feat_archetype", "")),
            ),
            axis=1,
        )
        notional_series = trade_df["_asset_entry_notional"].astype(float)
        total_notional = max(float(notional_series.sum()), 1e-12)
        selected_family_stats = {}
        for family in ("geo", "macro", "earnings", "other"):
            mask = family_series.eq(family)
            selected_family_stats[f"selected_{family}_exposure_share"] = round(
                float(notional_series[mask].sum()) / total_notional * 100.0,
                4,
            )
            selected_family_stats[f"selected_{family}_trades"] = int(mask.sum())

    skipped_geo = sum(
        1
        for row in allocation_rows
        if row.get("decision") == "skipped" and row.get("event_family") == "geo"
    )
    skipped_macro = sum(
        1
        for row in allocation_rows
        if row.get("decision") == "skipped" and row.get("event_family") == "macro"
    )

    stats = {
        "initial": round(initial, 2),
        "final": round(final_equity, 2),
        "total_return": round((final_equity / initial - 1.0) * 100.0, 4),
        "benchmark_return": round((final_passive_equity / initial - 1.0) * 100.0, 4),
        "excess_return": round((final_equity - final_passive_equity) / initial * 100.0, 4),
        "max_dd": round(max_dd, 4),
        "sharpe": round(strat_sharpe, 4),
        "sortino": round(strat_sortino, 4),
        "benchmark_sharpe": round(bench_sharpe, 4),
        "benchmark_sortino": round(bench_sortino, 4),
        "n_trades": int(len(trade_df)),
        "win_rate": round(win_rate, 4),
        "avg_pnl": round(avg_pnl, 4),
        "avg_gross_pnl": round(avg_gross_pnl, 4),
        "gross_trade_pnl": round(gross_trade_pnl, 2),
        "net_trade_pnl": round(net_trade_pnl, 2),
        "total_txn_cost": round(total_txn_cost, 2),
        "trade_txn_cost": round(trade_txn_cost, 2),
        "friction_fail_rate": round(friction_fail_rate, 4),
        "avg_position_size": round(float(position_sizes.mean() * 100.0), 4) if len(position_sizes) else 0.0,
        "median_position_size": round(float(np.median(position_sizes) * 100.0), 4) if len(position_sizes) else 0.0,
        "min_position_size": round(float(position_sizes.min() * 100.0), 4) if len(position_sizes) else 0.0,
        "max_position_size": round(float(position_sizes.max() * 100.0), 4) if len(position_sizes) else 0.0,
        "start_date": str(first_day.date()),
        "end_date": str(last_day.date()),
        "n_equity_days": int(len(equity_df)),
        "skip_max_concurrent": skip_max_concurrent,
        "skip_duplicate_symbol": skip_duplicate_symbol,
        "skip_insufficient_capital": skip_insufficient_capital,
        "skip_same_day_symbol_collapsed": skip_same_day_symbol_collapsed,
        "skip_preempt_hurdle": skip_preempt_hurdle,
        "preemptions": preemptions,
        **selected_family_stats,
        "skipped_high_priority_opportunities": int(skipped_geo + skipped_macro),
        "skipped_geo_opportunities": int(skipped_geo),
        "skipped_macro_opportunities": int(skipped_macro),
        "allocation_mode": allocation_mode,
    }
    allocation_df = pd.DataFrame(allocation_rows)
    disposition_df = pd.DataFrame(candidate_disposition_rows)
    return trade_df, equity_df, stats, policy, allocation_df, disposition_df


# ── CEM objective ────────────────────────────────────────────────────────────

def daily_equity_sharpe(equity_df: pd.DataFrame) -> float | None:
    """Annualized Sharpe from portfolio daily equity returns."""
    if equity_df.empty or len(equity_df) < MIN_DAILY_RETURNS_FOR_SHARPE + 1:
        return None

    daily_returns = equity_df["equity"].astype(float).pct_change().dropna()
    if len(daily_returns) < MIN_DAILY_RETURNS_FOR_SHARPE:
        return None

    std = float(daily_returns.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0

    sharpe = float(daily_returns.mean() / std * math.sqrt(252.0))
    return sharpe if np.isfinite(sharpe) else None


def cem_reward(trades: pd.DataFrame, equity_df: pd.DataFrame, stats: dict, use_hurdle: bool) -> float:
    """Cost-aware daily-equity objective for CEM."""
    if trades.empty or stats["n_trades"] < MIN_TRADES_FOR_REWARD:
        return INVALID_SCORE

    sharpe = daily_equity_sharpe(equity_df)
    if sharpe is None:
        return INVALID_SCORE

    score = sharpe - DD_PENALTY * abs(float(stats["max_dd"]))

    if use_hurdle:
        # Fully realised cost-aware penalty. This deliberately does NOT claim
        # to be a live entry gate; it selects policies whose realised trades
        # more often clear the friction multiple.
        failed = (trades["gross_pnl"] < HURDLE_MULT * trades["txn_cost"]).mean()
        score -= float(failed) * HURDLE_PENALTY

    return float(score)


def _cem_fit_policy(
    fit_df: pd.DataFrame,
    prices: dict,
    probs: dict,
    *,
    bench_sym: str,
    use_hurdle: bool,
    use_kelly: bool,
    allocation_mode: str,
    fit_cutoff: pd.Timestamp,
    fit_eval_end: pd.Timestamp,
    n_iter: int,
    pop: int,
    seed: int,
    phase_tag: str,
) -> tuple[dict, float, list[dict[str, Any]]]:
    """
    Fit a CEM policy on one information set.

    `fit_df` must already satisfy t_e < fit_cutoff. The independent
    fit_eval_end horizon then truncates market paths before the next block.

    Also returns the full fitness population -- one row per (iteration, member).
    The returned policy is the argmax over these; any statistic that corrects for
    the search (Deflated Sharpe, and the true trial count for SPA) needs the whole
    distribution, not just the winner.
    """
    assert_rows_completed_before(fit_df, fit_cutoff, context=phase_tag)

    rng = np.random.default_rng(seed)
    names = list(PORTFOLIO_BOUNDS.keys())
    dim = len(names)
    elite_count = max(2, int(pop * CEM_ELITE_FRAC))
    mean = np.array([PORT_DEFAULT[name] for name in names], dtype=float)
    std = np.array(
        [(PORTFOLIO_BOUNDS[name][1] - PORTFOLIO_BOUNDS[name][0]) / 4.0 for name in names],
        dtype=float,
    )

    fit_start, _ = _frame_bounds(fit_df)
    tags = (
        (f"[Friction={HURDLE_MULT:.0f}x]" if use_hurdle else "")
        + ("[Kelly]" if use_kelly else "")
    )

    best_score = -np.inf
    best_policy: dict | None = None
    population_rows: list[dict[str, Any]] = []

    for iteration in range(n_iter):
        samples = rng.normal(mean, std, size=(pop, dim))
        policies = [port_policy_from_vec(sample) for sample in samples]
        scores: list[float] = []

        for policy in policies:
            trades, equity, stats, _, _, _ = sim_opp_cost(
                fit_df,
                prices,
                probs,
                policy,
                bench_sym=bench_sym,
                initial=INITIAL_CAPITAL,
                use_kelly=use_kelly,
                start_date=fit_start,
                end_date=fit_eval_end,
                allocation_mode=allocation_mode,
            )
            scores.append(cem_reward(trades, equity, stats, use_hurdle))

        score_array = np.asarray(scores, dtype=float)
        elite_idx = np.argsort(score_array)[-elite_count:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-4

        iteration_best_idx = int(np.argmax(score_array))
        iteration_best_score = float(score_array[iteration_best_idx])
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_policy = policies[iteration_best_idx]

        elite_set = set(int(i) for i in elite_idx)
        for member, (member_score, member_policy) in enumerate(zip(scores, policies)):
            population_rows.append(
                {
                    "benchmark": bench_sym,
                    "phase_tag": phase_tag,
                    "seed": seed,
                    "iteration": iteration + 1,
                    "member": member,
                    "score": float(member_score),
                    "is_invalid": bool(member_score <= INVALID_SCORE / 2),
                    "is_elite": member in elite_set,
                    "position_size_pct": float(member_policy.get("position_size_pct", float("nan"))),
                    "max_concurrent": int(member_policy.get("max_concurrent", 0)),
                }
            )

        print(
            f"    {bench_sym}|{phase_tag}{tags} iter {iteration + 1}/{n_iter}  "
            f"best={iteration_best_score:+.3f}  global={best_score:+.3f}",
            flush=True,
        )

    if best_policy is None or best_score <= INVALID_SCORE / 2:
        raise RuntimeError(
            f"No valid CEM policy for {bench_sym} in {phase_tag}. "
            "Increase completed train data, reduce window strictness, or inspect candidate generation."
        )

    return best_policy, float(best_score), population_rows


def cem_search(
    train_split_df: pd.DataFrame,
    prices: dict,
    probs: dict,
    *,
    bench_sym: str,
    use_hurdle: bool,
    use_wf: bool,
    use_kelly: bool,
    allocation_mode: str,
    train_fit_cutoff: pd.Timestamp,
    n_iter: int = CEM_ITERS,
    pop: int = CEM_POP,
    seed: int = CEM_BASE_SEED,
) -> tuple[
    dict | Callable[[pd.Timestamp], dict],
    float,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Returns (policy, objective, fold_audits, cem_population)."""
    if train_split_df.empty:
        raise ValueError("CEM received an empty train frame.")

    train_fit_cutoff = as_utc_day(train_fit_cutoff)
    final_fit_df = rows_completed_before(train_split_df, train_fit_cutoff)
    assert_rows_completed_before(
        final_fit_df,
        train_fit_cutoff,
        context="final train fit",
    )
    final_fit_end = train_fit_cutoff - pd.Timedelta(days=1)

    if not use_wf:
        policy, train_score, population = _cem_fit_policy(
            final_fit_df,
            prices,
            probs,
            bench_sym=bench_sym,
            use_hurdle=use_hurdle,
            use_kelly=use_kelly,
            allocation_mode=allocation_mode,
            fit_cutoff=train_fit_cutoff,
            fit_eval_end=final_fit_end,
            n_iter=n_iter,
            pop=pop,
            seed=seed,
            phase_tag="TrainFull",
        )
        return policy, train_score, [], population

    max_t = as_utc_day(train_split_df["t_theta"].max()) + pd.Timedelta(days=1)
    folds = create_expanding_wf_folds(train_split_df, max_t)
    if len(folds) < WF_MIN_FOLDS:
        raise RuntimeError(
            "T2 needs at least "
            f"{WF_MIN_FOLDS} expanding label-complete folds, but only {len(folds)} "
            "could be formed."
        )

    fold_audits: list[dict[str, Any]] = []
    oof_scores: list[float] = []
    wf_policies: list[dict[str, Any]] = []
    population: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        fold_policy, fit_score, fold_population = _cem_fit_policy(
            fold["fit_df"],
            prices,
            probs,
            bench_sym=bench_sym,
            use_hurdle=use_hurdle,
            use_kelly=use_kelly,
            allocation_mode=allocation_mode,
            fit_cutoff=fold["fit_cutoff"],
            fit_eval_end=fold["fit_end"],
            n_iter=n_iter,
            pop=pop,
            seed=seed,
            phase_tag=f"WF-Fold{fold_id}",
        )
        population.extend(fold_population)

        eval_trades, eval_equity, eval_stats, _, _, _ = sim_opp_cost(
            fold["eval_df"],
            prices,
            probs,
            fold_policy,
            bench_sym=bench_sym,
            initial=INITIAL_CAPITAL,
            use_kelly=use_kelly,
            start_date=fold["eval_start"],
            end_date=fold["eval_end"],
            allocation_mode=allocation_mode,
        )
        eval_score = cem_reward(eval_trades, eval_equity, eval_stats, use_hurdle)
        if eval_score <= INVALID_SCORE / 2:
            print(
                f"    Warning: Walk-forward fold {fold_id} for {bench_sym} has no valid OOF "
                "portfolio score. The block may be too sparse for the current "
                "minimum-trade/Sharpe requirements.",
                flush=True,
            )

        fit_t_e = pd.to_datetime(fold["fit_df"]["t_e"], utc=True)
        fold_audits.append(
            {
                "fold": fold_id,
                "fit_start_date": str(fold["fit_start"].date()),
                "fit_label_cutoff": str(fold["fit_cutoff"].date()),
                "fit_eval_end_date": str(fold["fit_end"].date()),
                "fit_candidates": int(len(fold["fit_df"])),
                "fit_latest_t_e": str(as_utc_day(fit_t_e.max()).date()),
                "fit_cem_score": round(fit_score, 6),
                "eval_start_date": str(fold["eval_start"].date()),
                "eval_end_date": str(fold["eval_end"].date()),
                "eval_candidates": int(len(fold["eval_df"])),
                "eval_oof_score": round(eval_score, 6),
                "eval_return_pct": eval_stats["total_return"],
                "eval_benchmark_return_pct": eval_stats["benchmark_return"],
                "eval_excess_return_pct": eval_stats["excess_return"],
                "eval_max_dd_pct": eval_stats["max_dd"],
                "eval_trades": eval_stats["n_trades"],
                "eval_policy_json": json.dumps(fold_policy, sort_keys=True),
            }
        )
        oof_scores.append(float(eval_score))
        wf_policies.append(
            {
                "fold": fold_id,
                "eval_start": fold["eval_start"],
                "eval_end_exclusive": fold["eval_end_exclusive"],
                "policy": fold_policy,
            }
        )

        print(
            f"    {bench_sym}|WF-F{fold_id}  "
            f"fit_n={len(fold['fit_df'])} (t_e < {fold['fit_cutoff'].date()})  "
            f"eval_n={len(fold['eval_df'])} ({fold['eval_start'].date()}—"
            f"{fold['eval_end'].date()})  OOF={eval_score:+.3f}",
            flush=True,
        )

    return DynamicPolicySchedule(wf_policies), float(np.mean(oof_scores)), fold_audits, population


# ── Database loading ─────────────────────────────────────────────────────────

async def load_paths(df: pd.DataFrame) -> tuple[dict, dict]:
    """Load price/probability paths from the prebuilt pkl artifacts ONLY.

    The backtest is deliberately decoupled from Postgres: it reads exactly the
    three committed artifacts (candidates_audit_clean.parquet + prices.pkl + probs.pkl) and
    nothing else. The nightly `python -m ingest --rebuild` job regenerates them
    from the DB and pushes them. If they are missing we fail loudly rather than
    silently reaching for a live database that may have drifted.
    """
    import pickle
    from pathlib import Path
    data_dir = Path("data")
    prices_path = data_dir / "prices.pkl"
    probs_path = data_dir / "probs.pkl"
    if not (prices_path.exists() and probs_path.exists()):
        raise FileNotFoundError(
            f"Backtest requires prebuilt {prices_path} and {probs_path}. "
            "Regenerate them with `python -m ingest --rebuild`; the optimizer no "
            "longer reads Postgres directly."
        )
    print("  [Data] Loading cached prices and probabilities from data/*.pkl", flush=True)
    with open(prices_path, "rb") as f:
        prices = pickle.load(f)
    with open(probs_path, "rb") as f:
        probs = pickle.load(f)

    _CLOSE_CACHE.clear()
    _PATH_CUTOFF_CACHE.clear()
    clear_kernel_caches()
    return prices, probs


# ── Split, reporting, and audit output ───────────────────────────────────────

def split_train_val_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if "split" not in df.columns:
        raise ValueError("candidates.parquet must contain a chronological 'split' column.")

    split = df["split"].astype(str).str.lower().str.strip()
    train_df = df.loc[split == "train"].copy()
    val_df = df.loc[split == "val"].copy()
    test_df = df.loc[split == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Expected non-empty train/val/test splits; got "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}."
        )

    order_cols = [col for col in ("t_theta", "t_e", "market_id", "symbol") if col in df.columns]
    ordered_splits = df.sort_values(order_cols, kind="mergesort")["split"].astype(str).str.lower().tolist()
    seen_val = seen_test = False
    for sp in ordered_splits:
        if sp == "train":
            if seen_val or seen_test:
                raise ValueError("The split is not chronological by candidate order: train row appears after val/test.")
        elif sp == "val":
            seen_val = True
            if seen_test:
                raise ValueError("The split is not chronological by candidate order: val row appears after test.")
        elif sp == "test":
            seen_test = True

    val_start = as_utc_day(pd.to_datetime(val_df["t_theta"], utc=True).min())
    test_start = as_utc_day(pd.to_datetime(test_df["t_theta"], utc=True).min())
    overlapping_train = train_df[pd.to_datetime(train_df["t_theta"], utc=True) > val_start]
    overlapping_val = val_df[pd.to_datetime(val_df["t_theta"], utc=True) > test_start]
    if not overlapping_train.empty:
        raise ValueError(
            "The split is not chronological: some train candidates start after "
            "the first validation candidate. Rebuild candidates.parquet before trusting results."
        )
    if not overlapping_val.empty:
        raise ValueError(
            "The split is not chronological: some validation candidates start after "
            "the first test candidate. Rebuild candidates.parquet before trusting results."
        )

    return train_df, val_df, test_df, val_start, test_start


def save_audit_logs(
    *,
    experiment_label: str,
    benchmark: str,
    stage: str,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    allocation_df: pd.DataFrame | None = None,
    disposition_df: pd.DataFrame | None = None,
    candidate_df: pd.DataFrame | None = None,
    prices: dict | None = None,
    probs: dict | None = None,
    policy: dict | Callable[[pd.Timestamp], dict] | None = None,
    paths: OutputPaths | None = None,
) -> None:
    paths = paths or resolve_output_paths(None)
    paths.trade_log_dir.mkdir(parents=True, exist_ok=True)
    paths.equity_log_dir.mkdir(parents=True, exist_ok=True)
    paths.allocation_log_dir.mkdir(parents=True, exist_ok=True)
    paths.forensic_log_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{benchmark.lower()}_{_slug(experiment_label)}_{_slug(stage)}"
    trade_path = paths.trade_log_dir / f"{stem}.csv"
    equity_path = paths.equity_log_dir / f"{stem}.csv"
    allocation_path = paths.allocation_log_dir / f"{stem}.csv"
    trade_df.to_csv(trade_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    if allocation_df is not None:
        allocation_df.to_csv(allocation_path, index=False)
    if disposition_df is not None and not disposition_df.empty:
        disposition_path = paths.disposition_dir / f"candidate_disposition_{stem}.csv"
        disposition_path.parent.mkdir(parents=True, exist_ok=True)
        disposition_df.to_csv(disposition_path, index=False)

    if candidate_df is not None and prices is not None and probs is not None:
        write_trade_forensics(
            experiment_label=experiment_label,
            benchmark=benchmark,
            stage=stage,
            trade_df=trade_df,
            equity_df=equity_df,
            candidate_df=candidate_df,
            prices=prices,
            probs=probs,
            policy=policy,
            output_path=paths.forensic_log_dir / f"{stem}_forensics.csv",
            source_trade_log=trade_path,
            source_equity_log=equity_path,
        )


def _print_folds(folds: list[dict[str, Any]]) -> None:
    print(
        f"\n  Expanding label-complete walk-forward folds: {len(folds)} "
        "(fit eligibility: t_e < fold start)",
        flush=True,
    )
    for fold in folds:
        print(
            f"    F{fold['fold']}: fit={len(fold['fit_df'])} "
            f"(t_e < {fold['fit_cutoff'].date()})  →  "
            f"eval={len(fold['eval_df'])} "
            f"({fold['eval_start'].date()} — {fold['eval_end'].date()})",
            flush=True,
        )


def completed_trade_history_before(
    trades: pd.DataFrame,
    cutoff: Any,
) -> list[dict[str, Any]]:
    """
    Return realised trade history usable at a stage boundary.

    Kelly uses realised, net trade returns, but this boundary check is still
    conservative: a candidate may enter the history only when its associated
    outcome timestamp t_e is before the next stage begins.
    """
    if trades.empty:
        return []

    completed = trades.loc[
        trades["realized_exit_reason"] != "evaluation_end_liquidation"
    ].copy()
    if completed.empty:
        return []

    if "candidate_t_e" not in completed.columns:
        raise ValueError("Trade log is missing candidate_t_e required for Kelly eligibility.")

    cutoff_ts = _as_utc_timestamp(cutoff)
    completed_at = pd.to_datetime(completed["candidate_t_e"], utc=True)
    return completed.loc[completed_at < cutoff_ts].to_dict("records")


def _print_table(results: list[dict[str, Any]], *, prefix: str, label: str) -> None:
    header = (
        f"  {'Experiment':<20} {label + ' Ret':>9} {'B&H':>9} {'Excess':>9} {label + ' DD':>9} "
        f"{'Sharpe':>7} {'B&H Shp':>7} "
        f"{'Trades':>7} {'Win%':>7} {'AvgNet$':>10} {'TradeCost$':>12} "
        f"{'AvgPos':>8} {'MaxConc':>8} {'Sample':>8}"
    )
    print(header)
    print(f"  {'-' * 153}")

    for row in results:
        sample = "thin" if row[f"{prefix}_trades"] < EVAL_TRADE_WARNING_N else "ok"
        print(
            f"  {row['experiment']:<20} "
            f"{row[f'{prefix}_return_pct']:>+8.2f}% "
            f"{row[f'{prefix}_benchmark_return_pct']:>+8.2f}% "
            f"{row[f'{prefix}_excess_return_pct']:>+8.2f}% "
            f"{row[f'{prefix}_max_dd_pct']:>+8.2f}% "
            f"{row[f'{prefix}_sharpe']:>7.2f} "
            f"{row[f'{prefix}_benchmark_sharpe']:>7.2f} "
            f"{row[f'{prefix}_trades']:>6} "
            f"{row[f'{prefix}_win_rate_pct']:>6.1f}% "
            f"${row[f'{prefix}_avg_net_pnl']:>9.2f} "
            f"${row[f'{prefix}_trade_txn_cost']:>11.0f} "
            f"{row[f'{prefix}_avg_position_size_pct']:>7.1f}% "
            f"{row['policy_max_concurrent']:>7} "
            f"{sample:>8}",
            flush=True,
        )


def _stage_metrics(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_return_pct": stats["total_return"],
        f"{prefix}_benchmark_return_pct": stats["benchmark_return"],
        f"{prefix}_excess_return_pct": stats["excess_return"],
        f"{prefix}_max_dd_pct": stats["max_dd"],
        f"{prefix}_sharpe": stats.get("sharpe", 0.0),
        f"{prefix}_benchmark_sharpe": stats.get("benchmark_sharpe", 0.0),
        f"{prefix}_start_date": stats["start_date"],
        f"{prefix}_end_date": stats["end_date"],
        f"{prefix}_equity_days": stats["n_equity_days"],
        f"{prefix}_trades": stats["n_trades"],
        f"{prefix}_win_rate_pct": stats["win_rate"],
        f"{prefix}_avg_net_pnl": stats["avg_pnl"],
        f"{prefix}_avg_gross_pnl": stats["avg_gross_pnl"],
        f"{prefix}_gross_trade_pnl": stats["gross_trade_pnl"],
        f"{prefix}_net_trade_pnl": stats["net_trade_pnl"],
        f"{prefix}_total_txn_cost": stats["total_txn_cost"],
        f"{prefix}_trade_txn_cost": stats["trade_txn_cost"],
        f"{prefix}_friction_fail_rate_pct": stats["friction_fail_rate"],
        f"{prefix}_avg_position_size_pct": stats["avg_position_size"],
        f"{prefix}_median_position_size_pct": stats["median_position_size"],
        f"{prefix}_min_position_size_pct": stats["min_position_size"],
        f"{prefix}_max_position_size_pct": stats["max_position_size"],
        f"{prefix}_thin_sample": stats["n_trades"] < EVAL_TRADE_WARNING_N,
        f"{prefix}_skip_max_concurrent": stats.get("skip_max_concurrent", 0),
        f"{prefix}_skip_duplicate_symbol": stats.get("skip_duplicate_symbol", 0),
        f"{prefix}_skip_insufficient_capital": stats.get("skip_insufficient_capital", 0),
        f"{prefix}_skip_same_day_symbol_collapsed": stats.get("skip_same_day_symbol_collapsed", 0),
        f"{prefix}_skip_preempt_hurdle": stats.get("skip_preempt_hurdle", 0),
        f"{prefix}_preemptions": stats.get("preemptions", 0),
        f"{prefix}_selected_geo_exposure_share": stats.get("selected_geo_exposure_share", 0.0),
        f"{prefix}_selected_macro_exposure_share": stats.get("selected_macro_exposure_share", 0.0),
        f"{prefix}_selected_earnings_exposure_share": stats.get("selected_earnings_exposure_share", 0.0),
        f"{prefix}_selected_other_exposure_share": stats.get("selected_other_exposure_share", 0.0),
        f"{prefix}_selected_geo_trades": stats.get("selected_geo_trades", 0),
        f"{prefix}_selected_macro_trades": stats.get("selected_macro_trades", 0),
        f"{prefix}_selected_earnings_trades": stats.get("selected_earnings_trades", 0),
        f"{prefix}_selected_other_trades": stats.get("selected_other_trades", 0),
        f"{prefix}_skipped_high_priority_opportunities": stats.get("skipped_high_priority_opportunities", 0),
        f"{prefix}_skipped_geo_opportunities": stats.get("skipped_geo_opportunities", 0),
        f"{prefix}_skipped_macro_opportunities": stats.get("skipped_macro_opportunities", 0),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def _experiment_slug_map() -> dict[str, dict]:
    return {_slug(exp["label"]): exp for exp in ALL_EXPERIMENTS}


def _select_experiments(requested: list[str] | None) -> list[dict]:
    """`None` -> the active EXPERIMENTS list. Otherwise resolve slugs against
    ALL_EXPERIMENTS, so a parked arm can be run without editing source."""
    if not requested:
        return list(EXPERIMENTS)

    slug_map = _experiment_slug_map()
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(ALL_EXPERIMENTS)

    selected: list[dict] = []
    unknown: list[str] = []
    for name in requested:
        key = _slug(name)
        if key in slug_map:
            selected.append(slug_map[key])
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(
            f"Unknown experiment(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(sorted(slug_map))}, or 'all'."
        )
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CEM experiment matrix. Defaults reproduce the historical run exactly."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        metavar="SLUG",
        help=(
            "Experiment slugs to run (e.g. baseline t4_geopriority), or 'all'. "
            "Defaults to the active EXPERIMENTS list."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["SPY", "QQQ"],
        metavar="SYM",
        help="Benchmarks to run against. Default: SPY QQQ.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CEM_BASE_SEED,
        help=(
            f"CEM base seed (default {CEM_BASE_SEED}). The per-benchmark offset in "
            "BENCHMARK_SEED_OFFSET is added to this."
        ),
    )
    parser.add_argument(
        "--cem-iters",
        type=int,
        default=CEM_ITERS,
        help=f"CEM iterations per fit (default {CEM_ITERS}).",
    )
    parser.add_argument(
        "--cem-pop",
        type=int,
        default=CEM_POP,
        help=f"CEM population per iteration (default {CEM_POP}).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Namespace all outputs under runs/<run-id>/ so parallel or repeated runs "
            "do not clobber each other. Omit for the historical in-place layout."
        ),
    )
    parser.add_argument(
        "--dd-penalty",
        type=float,
        default=DD_PENALTY,
        help=(
            "Drawdown penalty lambda in the CEM reward "
            "Sharpe - lambda * abs(max_dd_percent)."
        ),
    )
    parser.add_argument(
        "--no-allocation-log",
        action="store_true",
        help="Skip per-candidate allocation/disposition logging (faster, smaller output).",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help=(
            "Candidate parquet to evaluate. Defaults to the audit-cleaned, "
            "primary-eligible artifact. Pass data/candidates.parquet only for an "
            "explicit legacy comparison."
        ),
    )
    parser.add_argument(
        "--no-forensics-log",
        action="store_true",
        help="Skip the derived trade-forensics CSV (faster, smaller output).",
    )
    args = parser.parse_args(argv)
    if args.cem_iters < 1 or args.cem_pop < 2:
        raise SystemExit("--cem-iters must be >= 1 and --cem-pop must be >= 2.")
    args.benchmarks = [b.upper() for b in args.benchmarks]
    unknown_bench = [b for b in args.benchmarks if b not in BENCHMARK_SEED_OFFSET]
    if unknown_bench:
        raise SystemExit(
            f"Unknown benchmark(s): {', '.join(unknown_bench)}. "
            f"Known: {', '.join(BENCHMARK_SEED_OFFSET)}."
        )
    return args


def main(argv: list[str] | None = None) -> None:
    global DD_PENALTY
    args = _parse_args(argv)
    DD_PENALTY = float(args.dd_penalty)
    experiments = _select_experiments(args.experiments)
    benchmarks = tuple(args.benchmarks)
    paths = resolve_output_paths(args.run_id)
    collect_allocation_log = not args.no_allocation_log
    collect_forensic_log = not args.no_forensics_log

    started = time.time()

    print("=" * 78)
    print(f"  CLEAN {len(experiments)}-EXPERIMENT COMPARISON x {len(benchmarks)} BENCHMARK(S)")
    print("  Label-complete CEM | Expanding walk-forward T2 | Fully net trade costs")
    print(f"  seed={args.seed}  run_id={args.run_id or '<in-place>'}  "
          f"cem_iters={args.cem_iters}  cem_pop={args.cem_pop}  "
          f"allocation_log={'on' if collect_allocation_log else 'off'}  "
          f"forensics_log={'on' if collect_forensic_log else 'off'}  "
          f"dd_penalty={DD_PENALTY:g}")
    print("=" * 78)

    candidates_path = args.candidates_path
    if not candidates_path.is_absolute():
        candidates_path = PROJECT / candidates_path
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Candidate artifact not found: {candidates_path}. "
            "Run `python -m ingest.clean_candidates` first."
        )
    df = pd.read_parquet(candidates_path)

    required_columns = {"symbol", "market_id", "t_theta", "t_e", "split", REL_COL}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{candidates_path} is missing required columns: {missing}")

    if "cem_eligible" in df.columns:
        ineligible = ~df["cem_eligible"].fillna(False).astype(bool)
        if ineligible.any():
            print(
                f"  [Data cleaning] filtering {int(ineligible.sum())} non-primary rows "
                f"from {candidates_path.name}",
                flush=True,
            )
            df = df.loc[~ineligible].copy()
    df = df[df[REL_COL].astype(float) > 0.5].copy()

    # `val` is not used for any CEM/model selection in this pipeline (the OOS set
    # is defined purely by t_theta >= oos_start, and the frozen policy is chosen
    # without consulting a validation split). Fold `val` into `test` so the trade
    # logs carry a single, contiguous out-of-sample label instead of an inert one.
    if "split" in df.columns:
        df["split"] = (
            df["split"].astype(str).str.lower().str.strip().replace({"val": "test"})
        )

    df["t_theta"] = pd.to_datetime(df["t_theta"], utc=True)
    df["t_e"] = pd.to_datetime(df["t_e"], utc=True)

    invalid_time_order = df["t_e"] < df["t_theta"]
    if invalid_time_order.any():
        sample = df.loc[invalid_time_order, ["symbol", "market_id", "t_theta", "t_e"]].head(5)
        raise ValueError(
            "Found candidate rows with t_e earlier than t_theta; cannot enforce "
            f"label-complete walk-forward training. Examples:\n{sample.to_string(index=False)}"
        )

    print(
        f"\n  {len(df)} relevance-filtered candidates loaded from {candidates_path.name}",
        flush=True,
    )
    max_t = as_utc_day(df["t_theta"].max()) + pd.Timedelta(days=1)
    preview_folds = create_expanding_wf_folds(df, max_t)
    if not preview_folds:
        raise ValueError("Not enough data to create any Walk-Forward folds.")
    
    oos_start = pd.Timestamp("2026-01-01", tz="UTC")
    oos_end = max_t - pd.Timedelta(days=1)

    train_df = rows_completed_before(df, oos_start)
    if train_df.empty:
        raise ValueError("No train rows have t_e before the OOS start. A label-complete policy cannot be fitted.")
    
    train_eval_end = oos_start - pd.Timedelta(days=1)
    
    oos_candidates_count = len(df[(df["t_theta"] >= oos_start) & (df["t_theta"] <= oos_end)])

    print(
        f"  Total Candidates = {len(df)}\n"
        f"  Walk-Forward OOS Timeline = {oos_start.date()} to {oos_end.date()}\n"
        f"  Initial Train Fit Cutoff = t_e < {oos_start.date()}\n"
        f"  Train Samples (t_e < {oos_start.date()}) = {len(train_df)}\n"
        f"  Test OOS Samples = {oos_candidates_count}",
        flush=True,
    )

    prices, probs = asyncio.run(load_paths(df))
    print(f"  {len(prices)} symbols, {len(probs)} markets loaded", flush=True)

    _print_folds(preview_folds)

    all_results: list[dict[str, Any]] = []
    fold_audit_rows: list[dict[str, Any]] = []
    cem_population_rows: list[dict[str, Any]] = []

    for experiment_number, experiment in enumerate(experiments, start=1):
        label = experiment["label"]
        use_hurdle = bool(experiment["hurdle"])
        use_wf = bool(experiment["wf"])
        use_kelly = bool(experiment["kelly"])
        allocation_mode = str(experiment.get("allocation_mode", ALLOCATION_FIFO))

        print(f"\n{'=' * 78}")
        print(f"  EXPERIMENT {experiment_number}/{len(experiments)}: {label}", flush=True)
        flags: list[str] = []
        if use_hurdle:
            flags.append(f"realised friction penalty={HURDLE_MULT:.0f}x")
        if use_wf:
            flags.append(f"expanding t_e-complete folds={len(preview_folds)}")
        if use_kelly:
            flags.append("half-Kelly sizing")
        if allocation_mode == ALLOCATION_EVENT_PRIORITY:
            flags.append("geo-first event-priority allocation")
        print(f"  Techniques: {', '.join(flags) if flags else 'none'}", flush=True)
        print(f"{'=' * 78}", flush=True)

        for benchmark in benchmarks:
            print(f"\n  [Train CEM search — {benchmark}]", flush=True)
            dynamic_policy, objective, fold_audits, cem_population = cem_search(
                df,
                prices,
                probs,
                bench_sym=benchmark,
                use_hurdle=use_hurdle,
                use_wf=use_wf,
                use_kelly=use_kelly,
                allocation_mode=allocation_mode,
                train_fit_cutoff=oos_start,
                n_iter=args.cem_iters,
                pop=args.cem_pop,
                seed=args.seed + BENCHMARK_SEED_OFFSET[benchmark],
            )
            for population_row in cem_population:
                cem_population_rows.append(
                    {
                        "experiment": label,
                        "allocation_mode": allocation_mode,
                        "base_seed": args.seed,
                        "cem_iters": args.cem_iters,
                        "cem_pop": args.cem_pop,
                        **population_row,
                    }
                )
            for fold_audit in fold_audits:
                fold_audit_rows.append(
                    {
                        "experiment": label,
                        "benchmark": benchmark,
                        "hurdle_realized_fitness_penalty": use_hurdle,
                        "kelly": use_kelly,
                        "allocation_mode": allocation_mode,
                        **fold_audit,
                    }
                )

            # This is only a training diagnostic to build kelly history.
            train_trades, train_equity, train_stats, _, train_allocation, train_disposition = sim_opp_cost(
                train_df,
                prices,
                probs,
                dynamic_policy,
                bench_sym=benchmark,
                initial=INITIAL_CAPITAL,
                use_kelly=use_kelly,
                start_date=as_utc_day(train_df["t_theta"].min()),
                end_date=train_eval_end,
                allocation_mode=allocation_mode,
                collect_allocation_log=collect_allocation_log,
            )

            kelly_train_history = (
                completed_trade_history_before(train_trades, oos_start)
                if use_kelly
                else None
            )

            # Persist the train-window trades too (2024-07 — 2025-12): they hold
            # the pre-2026 war/Fed/macro events used for behavioral analysis.
            save_audit_logs(
                experiment_label=label,
                benchmark=benchmark,
                stage="train",
                trade_df=train_trades,
                equity_df=train_equity,
                allocation_df=train_allocation,
                disposition_df=train_disposition,
                paths=paths,
            )

            print(f"\n  [Walk-Forward OOS sim — {benchmark}]", flush=True)
            oos_df = df[(df["t_theta"] >= oos_start) & (df["t_theta"] <= oos_end)].copy()
            oos_trades, oos_equity, oos_stats, _, oos_allocation, oos_disposition = sim_opp_cost(
                oos_df,
                prices,
                probs,
                dynamic_policy,
                bench_sym=benchmark,
                initial=INITIAL_CAPITAL,
                use_kelly=use_kelly,
                start_date=oos_start,
                end_date=oos_end,
                initial_kelly_history=kelly_train_history,
                allocation_mode=allocation_mode,
                collect_allocation_log=collect_allocation_log,
            )

            save_audit_logs(
                experiment_label=label,
                benchmark=benchmark,
                stage="test",
                trade_df=oos_trades,
                equity_df=oos_equity,
                allocation_df=oos_allocation,
                disposition_df=oos_disposition,
                candidate_df=oos_df if collect_forensic_log else None,
                prices=prices,
                probs=probs,
                policy=dynamic_policy,
                paths=paths,
            )
            
            # Since dynamic_policy might be callable, we get the first one for logging base ps
            logged_policy = dynamic_policy(oos_start) if callable(dynamic_policy) else dynamic_policy
            if isinstance(dynamic_policy, DynamicPolicySchedule):
                policy_payload: dict[str, Any] | list[dict[str, Any]] = dynamic_policy.as_records()
                policy_scope = "online_refit_schedule"
            else:
                policy_payload = logged_policy
                policy_scope = "frozen_policy"

            result = {
                "experiment": label,
                "benchmark": benchmark,
                "hurdle_realized_fitness_penalty": use_hurdle,
                "train_windows": use_wf,
                "kelly": use_kelly,
                "allocation_mode": allocation_mode,
                "dd_penalty": DD_PENALTY,
                "base_seed": args.seed,
                "cem_iters": args.cem_iters,
                "cem_pop": args.cem_pop,
                "cem_objective": round(objective, 6),
                "cem_objective_scope": "online_refit" if use_wf else "train_fit",
                "policy_scope": policy_scope,
                "wf_folds": len(fold_audits),
                "train_fit_label_cutoff": str(oos_start.date()),
                "train_fit_candidates": len(train_df),
                "policy_base_position_size_pct": round(float(logged_policy.get("position_size_pct", 0.1)) * 100.0, 4),
                "policy_max_concurrent": int(logged_policy.get("max_concurrent", 10)),
                "policy_json": json.dumps(policy_payload, sort_keys=True),
                "policy_snapshot_json": json.dumps(logged_policy, sort_keys=True),
            }
            result.update(_stage_metrics("train", train_stats))
            result.update(_stage_metrics("test", oos_stats))
            all_results.append(result)

            adv = _calc_advanced_metrics(oos_equity["equity"])
            bench_adv = _calc_advanced_metrics(oos_equity["benchmark_equity"])

            _print_monthly_metrics(oos_equity, label, benchmark)

            total_skipped = (
                oos_stats.get('skip_max_concurrent', 0)
                + oos_stats.get('skip_duplicate_symbol', 0)
                + oos_stats.get('skip_insufficient_capital', 0)
                + oos_stats.get('skip_same_day_symbol_collapsed', 0)
                + oos_stats.get('skip_preempt_hurdle', 0)
            )
            print(
                f"    -> TEST={oos_stats['total_return']:+.2f}%  "
                f"B&H={oos_stats['benchmark_return']:+.2f}%  "
                f"excess={oos_stats['excess_return']:+.2f}%  "
                f"max_dd={oos_stats['max_dd']:+.2f}%  "
                f"sharpe={adv['sharpe']:.2f} (B&H {bench_adv['sharpe']:.2f})  "
                f"sortino={adv['sortino']:.2f}  "
                f"trades={oos_stats['n_trades']}  "
                f"win%={oos_stats['win_rate']:.1f}%  "
                f"trade_cost=${oos_stats['trade_txn_cost']:.0f}  "
                f"avg_pos={oos_stats['avg_position_size']:.1f}%  "
                f"max_conc={int(logged_policy.get('max_concurrent', 10))}  "
                f"preemptions={oos_stats.get('preemptions', 0)}  "
                f"sample={'thin' if oos_stats['n_trades'] < EVAL_TRADE_WARNING_N else 'ok'}",
                flush=True,
            )
            print(
                f"    -> SKIPPED: {total_skipped} total  "
                f"(roster_full={oos_stats.get('skip_max_concurrent', 0)}  "
                f"dup_symbol={oos_stats.get('skip_duplicate_symbol', 0)}  "
                f"same_day_symbol={oos_stats.get('skip_same_day_symbol_collapsed', 0)}  "
                f"preempt_hurdle={oos_stats.get('skip_preempt_hurdle', 0)}  "
                f"no_capital={oos_stats.get('skip_insufficient_capital', 0)})",
                flush=True,
            )

            # Generate individual graph for this run
            slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            fig, ax = plt.subplots(figsize=(10, 5))
            dates = pd.to_datetime(oos_equity['date'])
            ax.plot(dates, oos_equity['equity'], label='Strategy Equity', color='#1f77b4', linewidth=2)
            ax.plot(dates, oos_equity['benchmark_equity'], label=f'{benchmark} (B&H)', color='#2ca02c', linestyle='--', linewidth=2)
            ax.set_title(f"OOS Equity: {label} ({benchmark}) | Sharpe: {adv['sharpe']:.2f} | MaxDD: {oos_stats['max_dd']:+.2f}%")
            ax.set_ylabel("Equity ($)")
            
            # Best/Worst trade annotations removed as requested
            
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.7)
            paths.plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(paths.plot_dir / f"cem_{slug}_{benchmark}_individual.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    if not all_results:
        raise RuntimeError("No experiment produced a result row; nothing to write.")

    for stage, stage_label in (("train", "TRAIN"), ("test", "FINAL TEST")):
        for benchmark in benchmarks:
            rows = [row for row in all_results if row["benchmark"] == benchmark]
            if not rows:
                continue
            print(f"\n{'=' * 78}")
            print(f"  {stage_label} RESULTS — {benchmark} benchmark")
            print(f"{'=' * 78}")
            _print_table(rows, prefix=stage, label=stage.capitalize())

    paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.results_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    if fold_audit_rows:
        pd.DataFrame(fold_audit_rows).to_csv(paths.wf_fold_audit_csv, index=False)

    if cem_population_rows:
        paths.cem_population_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cem_population_rows).to_csv(paths.cem_population_csv, index=False)

    combined_forensic_path = combine_forensic_csvs(
        paths.forensic_log_dir,
        paths.results_csv.parent / "experiment_forensics_clean.csv",
    )

    elapsed = time.time() - started
    print(f"\n  Clean results saved to: {paths.results_csv}")
    if fold_audit_rows:
        print(f"  Walk-forward fold audit saved to: {paths.wf_fold_audit_csv}")
    if cem_population_rows:
        print(
            f"  CEM fitness population ({len(cem_population_rows)} evaluations) "
            f"saved to: {paths.cem_population_csv}"
        )
    print(f"  Validation/test trade logs saved to: {paths.trade_log_dir}")
    print(f"  Validation/test equity logs saved to: {paths.equity_log_dir}")
    print(f"  Allocation decision logs saved to: {paths.allocation_log_dir}")
    if combined_forensic_path is not None:
        print(f"  Trade forensic logs saved to: {paths.forensic_log_dir}")
        print(f"  Combined trade forensic CSV saved to: {combined_forensic_path}")
    print(f"  Total elapsed: {elapsed / 60.0:.1f} min")


if __name__ == "__main__":
    main()
