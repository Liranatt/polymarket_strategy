"""Gap-aware daily-OHLC execution kernel for the CEM audit.

Execution semantics:
* Entry remains at the first eligible daily close, matching the canonical study.
* A stop active on trading day d may use only the peak confirmed through d-1.
* If Open[d] <= stop[d], a long stop-market fills at Open[d] (gap-through).
* Otherwise, if Low[d] <= stop[d], it fills at stop[d].
* High[d] may raise the peak only after the day's stop test, and therefore can
  affect stops from d+1 onward. This removes unknown high/low ordering without
  assuming a favourable intraday path.
* Probability exits are evaluated at the daily close after the intraday stop test.

This module is isolated under audit/. The production kernel is untouched.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from core.policy import RELEVANCE_COL
from core.polarity import effective_prob_surge, effective_probs, resolve_polarity

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def wrap(fn):
            return fn
        return wrap

_USE_KERNEL = HAVE_NUMBA and os.environ.get("SIM_KERNEL", "1") != "0"
_DAY_NS = 86_400_000_000_000
_OPEN_BY_SYMBOL: dict[str, dict[int, float]] = {}
_SYM_CACHE: dict[tuple[int, str], tuple] = {}
_MKT_CACHE: dict[tuple[int, str], tuple] = {}


def set_open_prices(opens: dict[str, Any]) -> None:
    normalized: dict[str, dict[int, float]] = {}
    for sym, values in opens.items():
        day_map: dict[int, float] = {}
        iterator = values.items() if isinstance(values, dict) else values
        for ts, value in iterator:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")
            day_map[int(stamp.normalize().value)] = float(value)
        normalized[str(sym).upper()] = day_map
    _OPEN_BY_SYMBOL.clear()
    _OPEN_BY_SYMBOL.update(normalized)
    clear_kernel_caches()


def clear_kernel_caches() -> None:
    _SYM_CACHE.clear()
    _MKT_CACHE.clear()


def _symbol_arrays(prices: dict, sym: str) -> tuple:
    key = (id(prices), sym)
    cached = _SYM_CACHE.get(key)
    if cached is not None:
        return cached
    bars = prices.get(sym, [])
    n = len(bars)
    value = np.empty(n, dtype=np.int64)
    norm = np.empty(n, dtype=np.int64)
    open_ = np.empty(n, dtype=np.float64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    close = np.empty(n, dtype=np.float64)
    day_map = _OPEN_BY_SYMBOL.get(str(sym).upper(), {})
    for i, bar in enumerate(bars):
        ts = bar[0] if isinstance(bar[0], pd.Timestamp) else pd.Timestamp(bar[0])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        value[i] = ts.value
        day_key = int(ts.normalize().value)
        norm[i] = day_key
        open_[i] = float(day_map.get(day_key, np.nan))
        high[i] = float(bar[1])
        low[i] = float(bar[2])
        close[i] = float(bar[3])
    cached = (value, norm, open_, high, low, close, bars)
    _SYM_CACHE[key] = cached
    return cached


def _market_arrays(probs: dict, mkt: str) -> tuple:
    key = (id(probs), mkt)
    cached = _MKT_CACHE.get(key)
    if cached is not None:
        return cached
    points = probs.get(mkt, [])
    m = len(points)
    pt_value = np.empty(m, dtype=np.int64)
    pval_raw = np.empty(m, dtype=np.float64)
    day_to_val: dict[int, float] = {}
    for i, point in enumerate(points):
        ts = point[0] if isinstance(point[0], pd.Timestamp) else pd.Timestamp(point[0])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        pt_value[i] = ts.value
        pval_raw[i] = float(point[1])
        day_to_val[int(ts.normalize().value)] = float(point[1])
    if day_to_val:
        day_uni = np.array(sorted(day_to_val), dtype=np.int64)
        pval_uni = np.array([day_to_val[d] for d in day_uni], dtype=np.float64)
    else:
        day_uni = np.empty(0, dtype=np.int64)
        pval_uni = np.empty(0, dtype=np.float64)
    cached = (pt_value, pval_raw, day_uni, pval_uni, points)
    _MKT_CACHE[key] = cached
    return cached


@njit(cache=True)
def _bisect_left(a, x):
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bisect_right(a, x):
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _scan(
    bar_value, bar_norm, bar_open, bar_high, bar_low, bar_close,
    pt_value, pval_raw, day_uni, pval_uni,
    window_lo_value, t_e_value, first_eligible_value, resolution_cut_value,
    enter_strong, enter_floor, hold_days, atr_mult, lock_activate, theta_out,
    p_surge, max_prob_surge, r_surge, max_price_runup,
):
    none = (0, -1, -1, -1, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    w_start = _bisect_left(bar_value, window_lo_value)
    w_end = _bisect_right(bar_value, t_e_value)
    if w_end - w_start < 2:
        return none

    e0 = _bisect_left(pt_value, first_eligible_value)
    if e0 >= pt_value.shape[0]:
        return none
    entry_pt_index = -1
    held = 0
    k = e0
    while k < pt_value.shape[0]:
        if pval_raw[k] >= enter_strong:
            entry_pt_index = k
            break
        if pval_raw[k] >= enter_floor:
            held += 1
            if held >= hold_days:
                entry_pt_index = k
                break
        else:
            held = 0
        k += 1
    if entry_pt_index < 0:
        return none

    if (p_surge == p_surge) and p_surge > max_prob_surge:
        return none
    if (r_surge == r_surge) and r_surge > max_price_runup:
        return none

    gi = _bisect_left(bar_value, pt_value[entry_pt_index])
    if gi < w_start:
        gi = w_start
    if gi >= w_end or w_end - gi < 2 or bar_value[gi] >= resolution_cut_value:
        return none
    hold_end = _bisect_left(bar_value, resolution_cut_value)
    if hold_end > w_end:
        hold_end = w_end
    if hold_end - gi < 2:
        return none
    entry_price = bar_close[gi]

    h_start = gi - 15
    if h_start < w_start:
        h_start = w_start
    tr_sum = 0.0
    cnt = 0
    j = h_start + 1
    while j <= gi:
        hh, ll, pc = bar_high[j], bar_low[j], bar_close[j - 1]
        tr = hh - ll
        d2 = abs(hh - pc)
        d3 = abs(ll - pc)
        if d2 > tr:
            tr = d2
        if d3 > tr:
            tr = d3
        tr_sum += tr
        cnt += 1
        j += 1
    if cnt < 1 or entry_price == 0.0:
        return none
    atr = tr_sum / cnt
    if atr == 0.0:
        return none
    atr_pct = atr / entry_price

    peak = 0.0
    gj = gi
    while gj < hold_end:
        i_rel = gj - gi
        oo, hh, ll, cc = bar_open[gj], bar_high[gj], bar_low[gj], bar_close[gj]
        ret_c = cc / entry_price - 1.0
        ret_h = hh / entry_price - 1.0
        ret_l = ll / entry_price - 1.0
        reason = 0
        hard_floor_pct = 0
        gap_flag = 0
        active_stop_price = 0.0

        if i_rel > 0:
            stop_dist = atr_mult * atr_pct
            stop_ret = peak - stop_dist
            stop_reason = 1
            if peak >= lock_activate:
                hard_floor_pct = int(np.floor(peak * 100.0 + 1e-9))
                hard_floor = hard_floor_pct / 100.0
                if hard_floor > stop_ret:
                    stop_ret = hard_floor
                    stop_reason = 2
            active_stop_price = entry_price * (1.0 + stop_ret)

            if ret_l <= stop_ret:
                reason = stop_reason
                if oo == oo:
                    if oo <= active_stop_price:
                        cc = oo
                        gap_flag = 1
                    else:
                        cc = active_stop_price
                else:
                    cc = ll
                    gap_flag = 2
                ret_c = cc / entry_price - 1.0
            else:
                pv = 1.0
                idx = _bisect_left(day_uni, bar_norm[gj])
                if idx < day_uni.shape[0] and day_uni[idx] == bar_norm[gj]:
                    pv = pval_uni[idx]
                if pv < theta_out:
                    reason = 3
                elif gj == hold_end - 1:
                    reason = 4

        if reason != 0:
            lo = 0.0
            first = 1
            k = gi
            while k <= gj:
                rl = bar_low[k] / entry_price - 1.0
                if first == 1 or rl < lo:
                    lo = rl
                    first = 0
                k += 1
            return (
                1, int(entry_pt_index), int(gi), int(gj), cc, int(reason),
                int(hard_floor_pct), peak, lo, ret_c, entry_price,
                int(gap_flag), active_stop_price,
            )

        if i_rel > 0 and ret_h > peak:
            peak = ret_h
        gj += 1

    last = hold_end - 1
    cc = bar_close[last]
    ret_c = cc / entry_price - 1.0
    lo = 0.0
    first = 1
    k = gi
    while k < hold_end:
        rl = bar_low[k] / entry_price - 1.0
        if first == 1 or rl < lo:
            lo = rl
            first = 0
        k += 1
    return (1, int(entry_pt_index), int(gi), int(last), cc, 5, 0,
            peak, lo, ret_c, entry_price, 0, 0.0)


def _base_result(row, mkt, sym, question, polarity, polarity_source, ent,
                 entry_ts, entry_price, exit_ts, exit_price, reason,
                 peak, trough, ret_c, gap_flag, stop_price, probs):
    mkt_probs = probs.get(mkt, [])
    converged = "YES" if mkt_probs and mkt_probs[-1][1] >= 0.5 else "NO" if mkt_probs else "UNKNOWN"
    return dict(
        market_id=mkt, symbol=sym, question=question,
        polarity=polarity, polarity_source=polarity_source,
        pct=round(ent[1], 3), converged=converged,
        asset_confidence=row.get("confidence_score"),
        question_confidence=row.get("feat_llm_confidence"),
        archetype=row.get("feat_archetype", ""),
        relevance=round(float(row.get(RELEVANCE_COL, 0)), 3),
        split=row.get("split", ""),
        entry_date=str(entry_ts.date()), entry_prob=round(ent[1], 3),
        entry_price=round(float(entry_price), 6),
        exit_date=str(exit_ts.date()), exit_price=round(float(exit_price), 6),
        exit_reason=reason,
        peak_pct=round(float(peak) * 100, 4),
        trough_pct=round(float(trough) * 100, 4),
        return_pct=round(float(ret_c) * 100, 6),
        execution_model="prior_day_peak_gap_aware_ohlc",
        stop_gap_fill=bool(gap_flag == 1),
        stop_missing_open_fallback=bool(gap_flag == 2),
        active_stop_price=round(float(stop_price), 6) if stop_price else None,
    )


def simulate_one(row, prices, probs, policy):
    sym, mkt = str(row["symbol"]), str(row["market_id"])
    question = str(row.get("question", ""))
    polarity, polarity_source = resolve_polarity(question, sym)
    if polarity == 0:
        return None
    probs_eff = effective_probs(probs, mkt, polarity)
    t_theta = pd.Timestamp(row["t_theta"])
    t_e = pd.Timestamp(row["t_e"])
    t_theta = t_theta.tz_localize("UTC") if t_theta.tzinfo is None else t_theta.tz_convert("UTC")
    t_e = t_e.tz_localize("UTC") if t_e.tzinfo is None else t_e.tz_convert("UTC")

    bar_value, bar_norm, bar_open, bar_high, bar_low, bar_close, bars = _symbol_arrays(prices, sym)
    if bar_value.shape[0] < 2:
        return None
    pt_value, pval_raw, day_uni, pval_uni, points = _market_arrays(probs_eff, mkt)
    if pt_value.shape[0] == 0:
        return None
    p_surge = effective_prob_surge(row, polarity)
    r_surge = row.get("feat_runup_since_t0")
    result = _scan(
        bar_value, bar_norm, bar_open, bar_high, bar_low, bar_close,
        pt_value, pval_raw, day_uni, pval_uni,
        np.int64(t_theta.value) - 30 * _DAY_NS,
        np.int64(t_e.value), np.int64(t_theta.normalize().value),
        np.int64((t_e - pd.Timedelta(days=1)).value),
        float(policy["enter_strong"]), float(policy["enter_floor"]),
        int(policy["hold_days"]), float(policy["atr_mult"]),
        float(policy["lock_activate"]), float(policy["theta_out"]),
        float(p_surge) if p_surge is not None else float("nan"),
        float(policy.get("max_prob_surge", 999.0)),
        float(r_surge) if r_surge is not None else float("nan"),
        float(policy.get("max_price_runup", 999.0)),
    )
    if result[0] == 0:
        return None
    (_status, epi, egi, xgi, exit_price, rc, floor_pct, peak, trough,
     ret_c, entry_price, gap_flag, stop_price) = result
    ent = (points[int(epi)][0], float(points[int(epi)][1]))
    if rc == 1:
        reason = f"trailing_{policy['atr_mult']:.1f}ATR"
    elif rc == 2:
        reason = f"profit_lock_{floor_pct}%"
    elif rc == 3:
        reason = f"poly<{policy['theta_out']}"
    elif rc == 4:
        reason = "resolution-1d"
    else:
        reason = "end_of_window"
    return _base_result(
        row, mkt, sym, question, polarity, polarity_source, ent,
        bars[int(egi)][0], entry_price, bars[int(xgi)][0], exit_price,
        reason, peak, trough, ret_c, gap_flag, stop_price, probs_eff,
    )
