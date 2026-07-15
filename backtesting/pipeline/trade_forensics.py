"""Wide trade-level forensic exports for CEM experiment runs.

The simulator's normal trade log answers "what trade closed and PnL".
This module answers the more painful question: "what was happening around
that trade, and was it part of the drawdown?"
"""
from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

import numpy as np
import pandas as pd


IRAN_RE = re.compile(
    r"\b(iran|iranian|tehran|hormuz|kharg|natanz|dimona)\b",
    re.IGNORECASE,
)

IRAN_MACRO_SOURCE_URLS = [
    "https://www.theguardian.com/business/2026/mar/01/oil-price-surge-iran-us-israel-strikes-markets",
    "https://www.lemonde.fr/en/economy/article/2026/03/14/kharg-island-iran-s-key-oil-export-hub_6751437_19.html",
    "https://en.wikipedia.org/wiki/List_of_attacks_during_the_2026_Iran_war",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _day(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _date_str(value: Any) -> str:
    ts = _day(value)
    return "" if ts is None else str(ts.date())


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _pct(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(value * 100.0, 4)


def _series_path(df: pd.DataFrame, value_col: str, max_items: int = 80) -> str:
    if df.empty or value_col not in df.columns:
        return ""
    parts: list[str] = []
    for _, row in df.head(max_items).iterrows():
        value = _float(row[value_col])
        if value is None:
            continue
        parts.append(f"{row['date'].date()}:{value:.4f}")
    suffix = "" if len(df) <= max_items else f";...+{len(df) - max_items}"
    return ";".join(parts) + suffix


def _bars_frame(prices: dict, symbol: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    symbol = str(symbol)
    if symbol in cache:
        return cache[symbol]

    rows = []
    for item in prices.get(symbol, []):
        if len(item) < 4:
            continue
        ts = _day(item[0])
        if ts is None:
            continue
        rows.append(
            {
                "date": ts,
                "high": float(item[1]),
                "low": float(item[2]),
                "close": float(item[3]),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cache[symbol] = frame
    return frame


def _prob_frame(probs: dict, market_id: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market_id = str(market_id)
    if market_id in cache:
        return cache[market_id]

    rows = []
    for item in probs.get(market_id, []):
        if len(item) < 2:
            continue
        ts = _day(item[0])
        if ts is None:
            continue
        rows.append({"date": ts, "prob": float(item[1])})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    cache[market_id] = frame
    return frame


def _value_on_or_before(frame: pd.DataFrame, day: pd.Timestamp | None, col: str) -> float | None:
    if frame.empty or day is None or col not in frame.columns:
        return None
    sub = frame.loc[frame["date"] <= day]
    if sub.empty:
        return None
    return _float(sub.iloc[-1][col])


def _value_on_or_after(frame: pd.DataFrame, day: pd.Timestamp | None, col: str) -> float | None:
    if frame.empty or day is None or col not in frame.columns:
        return None
    sub = frame.loc[frame["date"] >= day]
    if sub.empty:
        return None
    return _float(sub.iloc[0][col])


def _window(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if frame.empty or start is None or end is None:
        return frame.iloc[0:0].copy()
    return frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()


def _return_between(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> float | None:
    first = _value_on_or_before(frame, start, "close")
    last = _value_on_or_before(frame, end, "close")
    if first is None or last is None or abs(first) < 1e-12:
        return None
    return last / first - 1.0


def _price_metrics(
    prices: dict,
    symbol: str,
    entry_day: pd.Timestamp | None,
    exit_day: pd.Timestamp | None,
    entry_price: float | None,
    cache: dict[str, pd.DataFrame],
    *,
    prefix: str,
) -> dict[str, Any]:
    bars = _bars_frame(prices, symbol, cache)
    trade_window = _window(bars, entry_day, exit_day)
    out: dict[str, Any] = {
        f"{prefix}_bars_available": int(len(bars)),
        f"{prefix}_trade_bar_count": int(len(trade_window)),
        f"{prefix}_close_path": _series_path(trade_window, "close"),
    }

    for days in (5, 10, 20):
        start = None if entry_day is None else entry_day - pd.Timedelta(days=days)
        out[f"{prefix}_pre_{days}d_return_pct"] = _pct(_return_between(bars, start, entry_day))
    for days in (5, 10):
        end = None if exit_day is None else exit_day + pd.Timedelta(days=days)
        out[f"{prefix}_post_exit_{days}d_return_pct"] = _pct(_return_between(bars, exit_day, end))

    out[f"{prefix}_entry_close"] = _value_on_or_before(bars, entry_day, "close")
    out[f"{prefix}_exit_close"] = _value_on_or_before(bars, exit_day, "close")
    out[f"{prefix}_entry_to_exit_close_return_pct"] = _pct(_return_between(bars, entry_day, exit_day))

    if trade_window.empty or entry_price is None or abs(entry_price) < 1e-12:
        return out

    high_idx = trade_window["high"].astype(float).idxmax()
    low_idx = trade_window["low"].astype(float).idxmin()
    high = float(trade_window.loc[high_idx, "high"])
    low = float(trade_window.loc[low_idx, "low"])
    out.update(
        {
            f"{prefix}_max_high_during_trade": high,
            f"{prefix}_max_high_date": str(trade_window.loc[high_idx, "date"].date()),
            f"{prefix}_min_low_during_trade": low,
            f"{prefix}_min_low_date": str(trade_window.loc[low_idx, "date"].date()),
            f"{prefix}_max_favorable_excursion_pct": round((high / entry_price - 1.0) * 100.0, 4),
            f"{prefix}_max_adverse_excursion_pct": round((low / entry_price - 1.0) * 100.0, 4),
            f"{prefix}_close_to_low_gap_pct": round(
                (float(trade_window.iloc[-1]["close"]) / low - 1.0) * 100.0, 4
            )
            if low > 0
            else None,
        }
    )
    return out


def _prob_metrics(
    probs: dict,
    market_id: str,
    candidate_day: pd.Timestamp | None,
    entry_day: pd.Timestamp | None,
    exit_day: pd.Timestamp | None,
    cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    frame = _prob_frame(probs, market_id, cache)
    trade_window = _window(frame, entry_day, exit_day)
    out: dict[str, Any] = {
        "prob_points_available": int(len(frame)),
        "prob_trade_point_count": int(len(trade_window)),
        "prob_at_candidate_day": _value_on_or_after(frame, candidate_day, "prob"),
        "prob_at_entry_day": _value_on_or_after(frame, entry_day, "prob"),
        "prob_at_exit_day": _value_on_or_before(frame, exit_day, "prob"),
        "prob_path_during_trade": _series_path(trade_window, "prob"),
    }
    if not trade_window.empty:
        out.update(
            {
                "prob_min_during_trade": float(trade_window["prob"].min()),
                "prob_max_during_trade": float(trade_window["prob"].max()),
                "prob_entry_to_exit_change": (
                    _value_on_or_before(frame, exit_day, "prob") or 0.0
                )
                - (_value_on_or_after(frame, entry_day, "prob") or 0.0),
            }
        )
    return out


def _prep_candidates(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str], int]]:
    cand = candidate_df.copy()
    if cand.empty:
        return cand, {}
    cand["_market_id_key"] = cand["market_id"].astype(str)
    cand["_symbol_key"] = cand["symbol"].astype(str)
    cand["_theta_day_key"] = pd.to_datetime(cand["t_theta"], utc=True).dt.date.astype(str)
    lookup: dict[tuple[str, str, str], int] = {}
    for idx, row in cand.iterrows():
        key = (row["_market_id_key"], row["_symbol_key"], row["_theta_day_key"])
        lookup.setdefault(key, idx)
    return cand, lookup


def _candidate_for_trade(
    trade: pd.Series,
    candidates: pd.DataFrame,
    lookup: dict[tuple[str, str, str], int],
) -> pd.Series | None:
    if candidates.empty:
        return None
    theta_day = str(trade.get("candidate_t_theta") or trade.get("entry_date") or "")
    key = (str(trade.get("market_id", "")), str(trade.get("symbol", "")), theta_day)
    idx = lookup.get(key)
    if idx is not None:
        return candidates.loc[idx]

    fallback = candidates.loc[
        (candidates["_market_id_key"] == str(trade.get("market_id", "")))
        & (candidates["_symbol_key"] == str(trade.get("symbol", "")))
    ]
    if len(fallback) == 1:
        return fallback.iloc[0]
    return None


def _prep_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    eq = equity_df.copy()
    if eq.empty or "date" not in eq.columns:
        return pd.DataFrame()
    eq["date"] = pd.to_datetime(eq["date"], utc=True).dt.normalize()
    eq = eq.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
    if "benchmark_equity" in eq.columns:
        eq["benchmark_equity"] = pd.to_numeric(eq["benchmark_equity"], errors="coerce")
    eq["equity_peak_to_date"] = eq["equity"].cummax()
    eq["drawdown_pct"] = (eq["equity"] / eq["equity_peak_to_date"] - 1.0) * 100.0
    return eq


def _equity_context(equity: pd.DataFrame) -> dict[str, Any]:
    if equity.empty:
        return {}
    worst_idx = equity["drawdown_pct"].astype(float).idxmin()
    out = {
        "run_start_date": str(equity.iloc[0]["date"].date()),
        "run_end_date": str(equity.iloc[-1]["date"].date()),
        "run_start_equity": float(equity.iloc[0]["equity"]),
        "run_end_equity": float(equity.iloc[-1]["equity"]),
        "run_max_drawdown_pct": round(float(equity.loc[worst_idx, "drawdown_pct"]), 4),
        "run_max_drawdown_date": str(equity.loc[worst_idx, "date"].date()),
    }
    if "benchmark_equity" in equity.columns:
        out["run_start_benchmark_equity"] = float(equity.iloc[0]["benchmark_equity"])
        out["run_end_benchmark_equity"] = float(equity.iloc[-1]["benchmark_equity"])
    return out


def _equity_window_metrics(
    equity: pd.DataFrame,
    entry_day: pd.Timestamp | None,
    exit_day: pd.Timestamp | None,
    run_context: dict[str, Any],
) -> dict[str, Any]:
    if equity.empty:
        return {}
    win = _window(equity, entry_day, exit_day)
    out: dict[str, Any] = {
        "open_during_run_max_drawdown": False,
        "equity_path_during_trade": _series_path(win, "equity"),
    }
    if entry_day is not None and exit_day is not None and run_context.get("run_max_drawdown_date"):
        max_dd_day = _day(run_context["run_max_drawdown_date"])
        out["open_during_run_max_drawdown"] = bool(
            max_dd_day is not None and entry_day <= max_dd_day <= exit_day
        )
    if win.empty:
        return out

    worst_idx = win["drawdown_pct"].astype(float).idxmin()
    out.update(
        {
            "equity_at_trade_entry_or_next": float(win.iloc[0]["equity"]),
            "equity_at_trade_exit_or_prior": float(win.iloc[-1]["equity"]),
            "equity_delta_during_trade": round(float(win.iloc[-1]["equity"] - win.iloc[0]["equity"]), 2),
            "worst_portfolio_drawdown_during_trade_pct": round(float(win.loc[worst_idx, "drawdown_pct"]), 4),
            "worst_portfolio_drawdown_during_trade_date": str(win.loc[worst_idx, "date"].date()),
            "max_open_positions_during_trade": int(win["open_positions"].max())
            if "open_positions" in win.columns
            else None,
            "avg_open_positions_during_trade": round(float(win["open_positions"].mean()), 4)
            if "open_positions" in win.columns
            else None,
        }
    )
    if "benchmark_equity" in win.columns:
        out["benchmark_equity_delta_during_trade"] = round(
            float(win.iloc[-1]["benchmark_equity"] - win.iloc[0]["benchmark_equity"]),
            2,
        )
    return out


def _run_drawdown_mark_metrics(
    *,
    prices: dict,
    symbol: str,
    benchmark: str,
    entry_day: pd.Timestamp | None,
    exit_day: pd.Timestamp | None,
    entry_price: float | None,
    qty: float | None,
    run_context: dict[str, Any],
    price_cache: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    max_dd_day = _day(run_context.get("run_max_drawdown_date"))
    out: dict[str, Any] = {
        "is_open_on_run_max_drawdown_date": False,
        "asset_close_on_run_max_drawdown_date": None,
        "asset_unrealized_return_on_run_max_drawdown_pct": None,
        "asset_unrealized_gross_pnl_on_run_max_drawdown": None,
        "benchmark_return_entry_to_run_max_drawdown_pct": None,
    }
    if (
        max_dd_day is None
        or entry_day is None
        or exit_day is None
        or entry_price is None
        or abs(entry_price) < 1e-12
        or not (entry_day <= max_dd_day <= exit_day)
    ):
        return out

    out["is_open_on_run_max_drawdown_date"] = True
    asset_bars = _bars_frame(prices, symbol, price_cache)
    bench_bars = _bars_frame(prices, benchmark, price_cache)
    close_at_dd = _value_on_or_before(asset_bars, max_dd_day, "close")
    out["asset_close_on_run_max_drawdown_date"] = close_at_dd
    out["benchmark_return_entry_to_run_max_drawdown_pct"] = _pct(
        _return_between(bench_bars, entry_day, max_dd_day)
    )
    if close_at_dd is None:
        return out
    ret = close_at_dd / entry_price - 1.0
    out["asset_unrealized_return_on_run_max_drawdown_pct"] = round(ret * 100.0, 4)
    out["asset_unrealized_gross_pnl_on_run_max_drawdown"] = (
        round((close_at_dd - entry_price) * qty, 2) if qty is not None else None
    )
    return out


def _policy_for(policy: dict | list | Callable[[pd.Timestamp], dict] | None, day: pd.Timestamp | None) -> dict:
    if policy is None:
        return {}
    if callable(policy):
        try:
            return dict(policy(day or pd.Timestamp("1970-01-01", tz="UTC")))
        except Exception:
            return {}
    if isinstance(policy, list):
        # Walk-forward schedule as stored in results-CSV policy_json:
        # [{fold, eval_start_date, eval_end_exclusive_date, policy}, ...]
        windows = sorted(policy, key=lambda e: str(e.get("eval_start_date", "")))
        if not windows:
            return {}
        day_ts = pd.Timestamp(day) if day is not None else pd.Timestamp("1970-01-01")
        if day_ts.tzinfo is None:
            day_ts = day_ts.tz_localize("UTC")
        matched = dict(windows[0].get("policy", {}))
        for entry in windows:
            start = pd.Timestamp(entry["eval_start_date"], tz="UTC")
            end_exclusive = pd.Timestamp(entry["eval_end_exclusive_date"], tz="UTC")
            if start <= day_ts < end_exclusive:
                return dict(entry.get("policy", {}))
            if day_ts >= start:
                matched = dict(entry.get("policy", {}))
        return matched
    return dict(policy)


def _external_urls(symbol: str, question: str, entry_date: str) -> dict[str, str]:
    symbol_q = quote_plus(str(symbol))
    query = quote_plus(f"{symbol} stock {entry_date} {question}")
    event_query = quote_plus(f"{question} {entry_date}")
    return {
        "yahoo_finance_history_url": f"https://finance.yahoo.com/quote/{symbol_q}/history/",
        "marketwatch_quote_url": f"https://www.marketwatch.com/investing/fund/{symbol_q}",
        "stock_behavior_search_url": f"https://duckduckgo.com/?q={query}",
        "event_context_search_url": f"https://duckduckgo.com/?q={event_query}",
    }


def write_trade_forensics(
    *,
    experiment_label: str,
    benchmark: str,
    stage: str,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    prices: dict,
    probs: dict,
    policy: dict | Callable[[pd.Timestamp], dict] | None,
    output_path: Path,
    source_trade_log: Path | None = None,
    source_equity_log: Path | None = None,
) -> Path:
    """Write one wide forensic CSV for a single experiment/benchmark/stage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates, candidate_lookup = _prep_candidates(candidate_df)
    equity = _prep_equity(equity_df)
    run_context = _equity_context(equity)
    price_cache: dict[str, pd.DataFrame] = {}
    prob_cache: dict[str, pd.DataFrame] = {}

    if trade_df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return output_path

    trades = trade_df.copy()
    trades["_entry_sort"] = pd.to_datetime(trades["entry_date"], utc=True)
    trades["_exit_sort"] = pd.to_datetime(trades["exit_date"], utc=True)
    trades = trades.sort_values(["_entry_sort", "_exit_sort", "market_id", "symbol"]).reset_index(drop=True)
    total_net_pnl = float(pd.to_numeric(trades.get("pnl", 0), errors="coerce").fillna(0).sum())

    records: list[dict[str, Any]] = []
    cumulative_pnl = 0.0
    for sequence, trade in trades.iterrows():
        entry_day = _day(trade.get("entry_date"))
        exit_day = _day(trade.get("exit_date"))
        candidate_day = _day(trade.get("candidate_t_theta") or trade.get("entry_date"))
        candidate_end_day = _day(trade.get("candidate_t_e"))
        candidate = _candidate_for_trade(trade, candidates, candidate_lookup)
        question = str(trade.get("question", ""))
        symbol = str(trade.get("symbol", ""))
        market_id = str(trade.get("market_id", ""))
        is_iran = bool(IRAN_RE.search(question))
        entry_month = "" if entry_day is None else entry_day.strftime("%Y-%m")
        is_feb_mar_iran = bool(is_iran and entry_month in {"2026-02", "2026-03"})
        pnl = _float(trade.get("pnl")) or 0.0
        cumulative_pnl += pnl
        current_policy = _policy_for(policy, candidate_day)

        row: dict[str, Any] = {
            "experiment": experiment_label,
            "benchmark": benchmark,
            "stage": stage,
            "trade_sequence_by_entry": int(sequence + 1),
            "source_trade_log": str(source_trade_log or ""),
            "source_equity_log": str(source_equity_log or ""),
            "entry_month": entry_month,
            "is_iran_question": is_iran,
            "is_feb_mar_iran_question": is_feb_mar_iran,
            "macro_context_label": "Iran war / Hormuz oil-risk window" if is_feb_mar_iran else "",
            "macro_context_source_urls": ";".join(IRAN_MACRO_SOURCE_URLS) if is_feb_mar_iran else "",
            "entry_delay_days_from_candidate": (
                int((entry_day - candidate_day).days)
                if entry_day is not None and candidate_day is not None
                else None
            ),
            "days_to_resolution_at_entry": (
                int((candidate_end_day - entry_day).days)
                if candidate_end_day is not None and entry_day is not None
                else None
            ),
            "trade_duration_calendar_days": (
                int((exit_day - entry_day).days)
                if entry_day is not None and exit_day is not None
                else None
            ),
            "run_total_net_trade_pnl": round(total_net_pnl, 2),
            "pnl_share_of_run_net_trade_pnl_pct": round(pnl / total_net_pnl * 100.0, 4)
            if abs(total_net_pnl) > 1e-12
            else None,
            "cumulative_net_trade_pnl_by_entry": round(cumulative_pnl, 2),
        }
        row.update(run_context)
        row.update(_external_urls(symbol, question, str(trade.get("entry_date", ""))))

        for key, value in current_policy.items():
            row[f"policy_{key}"] = value
        row["policy_json_at_candidate"] = json.dumps(current_policy, sort_keys=True) if current_policy else ""

        for col in trade_df.columns:
            if col.startswith("_entry_sort") or col.startswith("_exit_sort"):
                continue
            row[col] = trade.get(col)

        if candidate is not None:
            for col in candidate.index:
                if col.startswith("_"):
                    continue
                value = candidate[col]
                row[f"candidate_{col}"] = value.isoformat() if hasattr(value, "isoformat") else value
        else:
            row["candidate_lookup_missing"] = True

        entry_price = _float(trade.get("entry_price"))
        row.update(
            _price_metrics(
                prices,
                symbol,
                entry_day,
                exit_day,
                entry_price,
                price_cache,
                prefix="asset",
            )
        )
        row.update(
            _price_metrics(
                prices,
                benchmark,
                entry_day,
                exit_day,
                _value_on_or_before(_bars_frame(prices, benchmark, price_cache), entry_day, "close"),
                price_cache,
                prefix="benchmark",
            )
        )
        asset_ret = _float(row.get("asset_entry_to_exit_close_return_pct"))
        bench_ret = _float(row.get("benchmark_entry_to_exit_close_return_pct"))
        row["asset_minus_benchmark_close_return_pct"] = (
            round(asset_ret - bench_ret, 4) if asset_ret is not None and bench_ret is not None else None
        )
        row.update(_prob_metrics(probs, market_id, candidate_day, entry_day, exit_day, prob_cache))
        row.update(_equity_window_metrics(equity, entry_day, exit_day, run_context))
        row.update(
            _run_drawdown_mark_metrics(
                prices=prices,
                symbol=symbol,
                benchmark=benchmark,
                entry_day=entry_day,
                exit_day=exit_day,
                entry_price=entry_price,
                qty=_float(trade.get("_qty")),
                run_context=run_context,
                price_cache=price_cache,
            )
        )
        records.append(row)

    forensic = pd.DataFrame(records)
    if not forensic.empty and "pnl" in forensic.columns:
        pnl_series = pd.to_numeric(forensic["pnl"], errors="coerce")
        forensic["pnl_rank_worst_in_run"] = pnl_series.rank(method="min", ascending=True)
        forensic["pnl_rank_best_in_run"] = pnl_series.rank(method="min", ascending=False)
        forensic["negative_pnl_and_open_during_run_max_drawdown"] = (
            (pnl_series < 0) & forensic["is_open_on_run_max_drawdown_date"].astype(bool)
        )

    forensic.to_csv(output_path, index=False)
    return output_path


def combine_forensic_csvs(forensic_dir: Path, output_path: Path) -> Path | None:
    paths = sorted(forensic_dir.glob("*_forensics.csv"))
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 1]
    if not frames:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output_path, index=False)
    return output_path


def write_existing_trade_forensics(
    *,
    data_dir: Path = Path("data"),
    trade_log_dir: Path | None = None,
    equity_log_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Regenerate forensic CSVs from already-written experiment logs."""
    trade_log_dir = trade_log_dir or data_dir / "experiment_trade_logs_clean"
    equity_log_dir = equity_log_dir or data_dir / "experiment_equity_logs_clean"
    output_dir = output_dir or data_dir / "experiment_forensics_clean"

    candidates = pd.read_parquet(data_dir / "candidates.parquet")
    with open(data_dir / "prices.pkl", "rb") as handle:
        prices = pickle.load(handle)
    with open(data_dir / "probs.pkl", "rb") as handle:
        probs = pickle.load(handle)

    policy_lookup: dict[tuple[str, str], dict] = {}
    label_lookup: dict[tuple[str, str], str] = {}
    results_path = data_dir / "experiment_results_clean.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        for _, row in results.iterrows():
            try:
                policy = json.loads(row.get("policy_json", "{}"))
            except Exception:
                policy = {}
            key = (str(row["benchmark"]).upper(), _slug(row["experiment"]))
            policy_lookup[key] = policy
            label_lookup[key] = str(row["experiment"])

    outputs: list[Path] = []
    for trade_path in sorted(trade_log_dir.glob("*_test.csv")):
        stem = trade_path.stem
        benchmark, rest = stem.split("_", 1)
        experiment_slug = rest.removesuffix("_test")
        equity_path = equity_log_dir / trade_path.name
        if not equity_path.exists():
            continue
        trade_df = pd.read_csv(trade_path)
        equity_df = pd.read_csv(equity_path)
        lookup_key = (benchmark.upper(), experiment_slug)
        policy = policy_lookup.get(lookup_key, {})
        experiment_label = label_lookup.get(lookup_key, experiment_slug)
        output_path = output_dir / f"{stem}_forensics.csv"
        write_trade_forensics(
            experiment_label=experiment_label,
            benchmark=benchmark.upper(),
            stage="test",
            trade_df=trade_df,
            equity_df=equity_df,
            candidate_df=candidates,
            prices=prices,
            probs=probs,
            policy=policy,
            output_path=output_path,
            source_trade_log=trade_path,
            source_equity_log=equity_path,
        )
        outputs.append(output_path)

    combined = combine_forensic_csvs(output_dir, data_dir / "experiment_forensics_clean.csv")
    if combined is not None:
        outputs.append(combined)
    return outputs
