"""The one authoritative entry/exit kernel, shared by both planes.

Two implementations of the *same* trade semantics live here so they can never
drift apart:

* ``scan_candidate`` — numba-compiled fast path. The CEM backtest hot loop
  (``backtesting.optimize_cem``) drives this thousands of times per search, so
  it must stay JIT-compiled. Do not downgrade it to Python.
* ``_simulate_one_py`` — the pure-Python reference. It is the authoritative
  definition of the trade semantics; the live trader (``live.strategy_engine``)
  uses its explicit raw-signal mode because polarity remains research-only
  until it has been validated.

``simulate_one`` dispatches to whichever is active (``_USE_KERNEL``). The
``SIM_KERNEL=0`` parity test asserts the two paths return byte-identical dicts.

The numba section was formerly ``backtesting.pipeline.sim_kernel``; the reference
+ dispatch were formerly in ``backtesting.pipeline.strategy``. The book is
long-only in equities: entry fires on HIGH *effective* probability, where
``core.polarity`` decides per (question, symbol) pair whether the effective
path is raw P(YES) (+1), the flipped 1-P(YES) (-1, e.g. "war ends" markets for
USO), or no clean side at all (0 — the pair is skipped). There is no shorting.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core.policy import RELEVANCE_COL
from core.polarity import (
    clear_effective_probs_cache,
    effective_prob_surge,
    effective_probs,
    resolve_polarity,
)

try:  # numba is optional; callers fall back to pure Python when it is absent.
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is missing
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator so the module imports without numba installed."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


# The numba kernel is used automatically when numba is importable. It produces
# output identical to the pure-Python reference (_simulate_one_py); set
# SIM_KERNEL=0 to force the reference path (used by the parity test).
_USE_KERNEL = HAVE_NUMBA and os.environ.get("SIM_KERNEL", "1") != "0"


# ══════════════════════════════════════════════════════════════════════════════
# Numba-accelerated kernel (formerly pipeline/sim_kernel.py)
# ══════════════════════════════════════════════════════════════════════════════
#
# The kernel only ever sees int64/float64 numpy arrays and scalars and performs
# exactly the same arithmetic, in the same order, as the reference Python. All
# timestamp normalization and every date/reason string is produced in Python
# from the original tz-aware Timestamps, so the result is tz-correct and matches
# the pure-Python path field for field. It assumes chronologically sorted daily
# bars and probability points, which every loader guarantees.

# Caches keyed by (id(container), key). The container ids stay valid because the
# runners (and ``truncate_paths``) keep the price/probability dicts alive for the
# whole run, so repeated horizons reuse their arrays.
_SYM_CACHE: dict[tuple[int, str], tuple] = {}
_MKT_CACHE: dict[tuple[int, str], tuple] = {}


def clear_caches() -> None:
    """Drop cached numpy views (call when fresh price/probability dicts load)."""
    _SYM_CACHE.clear()
    _MKT_CACHE.clear()


def _symbol_arrays(prices: dict, sym: str) -> tuple:
    """Return cached ``(value, norm, high, low, close, bars)`` arrays for sym.

    ``value`` is each bar's epoch-ns (``Timestamp.value`` is tz-independent, so
    chronological comparisons match the original Timestamp comparisons). ``norm``
    is each bar's ``.normalize().value`` — computed with real Timestamps so the
    day key matches ``Timestamp.normalize()`` in any timezone. ``bars`` is the
    original list, kept so the caller can format exit dates from real Timestamps.
    """
    key = (id(prices), sym)
    cached = _SYM_CACHE.get(key)
    if cached is not None:
        return cached

    bars = prices.get(sym, [])
    n = len(bars)
    value = np.empty(n, dtype=np.int64)
    norm = np.empty(n, dtype=np.int64)
    high = np.empty(n, dtype=np.float64)
    low = np.empty(n, dtype=np.float64)
    close = np.empty(n, dtype=np.float64)
    for i in range(n):
        bar = bars[i]
        ts = bar[0]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        value[i] = ts.value
        norm[i] = ts.normalize().value
        high[i] = bar[1]
        low[i] = bar[2]
        close[i] = bar[3]

    cached = (value, norm, high, low, close, bars)
    _SYM_CACHE[key] = cached
    return cached


def _market_arrays(probs: dict, mkt: str) -> tuple:
    """Return cached arrays for a market's probability path.

    ``pt_value`` / ``pval_raw`` preserve the raw point order used by the entry
    rule. ``day_uni`` / ``pval_uni`` collapse to one value per normalized day
    (last point wins, matching the ``{t.normalize(): v}`` dict in the reference)
    for the O(log n) exit-day lookup. ``points`` is kept for entry-date strings.
    """
    key = (id(probs), mkt)
    cached = _MKT_CACHE.get(key)
    if cached is not None:
        return cached

    points = probs.get(mkt, [])
    m = len(points)
    pt_value = np.empty(m, dtype=np.int64)
    pval_raw = np.empty(m, dtype=np.float64)
    day_to_val: dict[int, float] = {}
    for i in range(m):
        point = points[i]
        ts = point[0]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        pt_value[i] = ts.value
        value = float(point[1])
        pval_raw[i] = value
        day_to_val[ts.normalize().value] = value  # last point of a day wins

    if day_to_val:
        day_uni = np.array(sorted(day_to_val.keys()), dtype=np.int64)
        pval_uni = np.array([day_to_val[d] for d in day_uni], dtype=np.float64)
    else:
        day_uni = np.empty(0, dtype=np.int64)
        pval_uni = np.empty(0, dtype=np.float64)

    cached = (pt_value, pval_raw, day_uni, pval_uni, points)
    _MKT_CACHE[key] = cached
    return cached


@njit(cache=True)
def _bisect_left(a, x):
    """First index ``i`` with ``a[i] >= x`` (``a`` ascending)."""
    lo = 0
    hi = a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bisect_right(a, x):
    """First index ``i`` with ``a[i] > x`` (``a`` ascending)."""
    lo = 0
    hi = a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# Reason codes returned by the kernel (mapped to strings by the caller):
#   0 none, 1 trailing-ATR, 2 profit-lock, 3 probability-out, 4 resolution-1d,
#   5 end-of-window.
@njit(cache=True)
def _scan(
    bar_value,
    bar_norm,
    bar_high,
    bar_low,
    bar_close,
    pt_value,
    pval_raw,
    day_uni,
    pval_uni,
    window_lo_value,
    t_e_value,
    first_eligible_value,
    resolution_cut_value,
    is_earnings,
    enter_strong,
    enter_floor,
    hold_days,
    atr_mult,
    lock_activate,
    theta_out,
    p_surge,
    max_prob_surge,
    r_surge,
    max_price_runup,
):
    """Reproduce simulate_one's numeric core; see module docstring for parity.

    Returns an 11-tuple
    ``(status, entry_pt_index, entry_gindex, exit_gindex, exit_price,
    reason_code, hard_floor_pct, peak, trough, ret_c, entry_price)``. ``status``
    is 0 when no trade is produced.
    """
    none = (0, -1, -1, -1, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    # Price window: bars with window_lo_value <= t <= t_e_value (both inclusive).
    w_start = _bisect_left(bar_value, window_lo_value)
    w_end = _bisect_right(bar_value, t_e_value)  # exclusive
    if w_end - w_start < 2:
        return none

    # Entry rule (entry_day): scan raw points whose timestamp >= the first
    # eligible day. Strong entry can fire on any point; otherwise enter_floor
    # must hold for hold_days consecutive points.
    m = pt_value.shape[0]
    e0 = _bisect_left(pt_value, first_eligible_value)
    if e0 >= m:
        return none

    entry_pt_index = -1
    held = 0
    k = e0
    while k < m:
        if pval_raw[k] >= enter_strong:
            entry_pt_index = k
            break
        elif pval_raw[k] >= enter_floor:
            held += 1
            if held >= hold_days:
                entry_pt_index = k
                break
        else:
            held = 0
        k += 1
    if entry_pt_index < 0:
        return none
    entry_ts_value = pt_value[entry_pt_index]

    # Prob-surge / price-runup gates (NaN means "field absent" -> skip).
    if (p_surge == p_surge) and (p_surge > max_prob_surge):
        return none
    if (r_surge == r_surge) and (r_surge > max_price_runup):
        return none

    # First window bar at/after the entry timestamp.
    gi = _bisect_left(bar_value, entry_ts_value)
    if gi < w_start:
        gi = w_start
    if gi >= w_end:
        return none
    if w_end - gi < 2:  # path needs at least two bars
        return none
    if bar_value[gi] >= resolution_cut_value:
        return none
    hold_end = _bisect_left(bar_value, resolution_cut_value)
    if hold_end > w_end:
        hold_end = w_end
    if hold_end - gi < 2:
        return none
    entry_price = bar_close[gi]

    # ATR over the <=16 bars ending at the entry bar (calc_atr): start index is
    # max(window_start, entry - 15), matching win[max(0, entry_idx - 15):...].
    h_start = gi - 15
    if h_start < w_start:
        h_start = w_start
    tr_sum = 0.0
    cnt = 0
    j = h_start + 1
    while j <= gi:
        hh = bar_high[j]
        ll = bar_low[j]
        pc = bar_close[j - 1]
        tr = hh - ll
        d2 = hh - pc
        if d2 < 0.0:
            d2 = -d2
        if d2 > tr:
            tr = d2
        d3 = ll - pc
        if d3 < 0.0:
            d3 = -d3
        if d3 > tr:
            tr = d3
        tr_sum += tr
        cnt += 1
        j += 1
    if cnt < 1:
        return none
    atr = tr_sum / cnt
    if atr == 0.0 or entry_price == 0.0:
        return none
    atr_pct = atr / entry_price

    # Exit scan across the holding path.
    peak = 0.0
    gj = gi
    while gj < hold_end:
        i_rel = gj - gi
        hh = bar_high[gj]
        ll = bar_low[gj]
        cc = bar_close[gj]
        ret_c = cc / entry_price - 1.0
        ret_h = hh / entry_price - 1.0
        ret_l = ll / entry_price - 1.0

        reason = 0
        hard_floor_pct = 0
        if i_rel > 0:
            stop_dist = atr_mult * atr_pct

            pv = 1.0
            idx = _bisect_left(day_uni, bar_norm[gj])
            if idx < day_uni.shape[0] and day_uni[idx] == bar_norm[gj]:
                pv = pval_uni[idx]
            if pv < theta_out:
                reason = 3
            elif ret_l <= peak - stop_dist:
                reason = 1
                cand = entry_price * (1.0 + peak - stop_dist)
                cc = ll if ll > cand else cand
                ret_c = cc / entry_price - 1.0
            elif peak >= lock_activate:
                hard_floor_pct = int(peak * 100.0)
                hard_floor = hard_floor_pct / 100.0
                if ret_l < hard_floor:
                    reason = 2
                    cand = entry_price * (1.0 + hard_floor)
                    cc = ll if ll > cand else cand
                    ret_c = cc / entry_price - 1.0
            if reason == 0:
                if gj == hold_end - 1:
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
                1,
                int(entry_pt_index),
                int(gi),
                int(gj),
                cc,
                int(reason),
                int(hard_floor_pct),
                peak,
                lo,
                ret_c,
                entry_price,
            )

        if i_rel == 0:
            peak = 0.0
        elif ret_h > peak:
            peak = ret_h
        gj += 1

    # No exit triggered: liquidate on the last eligible pre-resolution bar.
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
    return (
        1,
        int(entry_pt_index),
        int(gi),
        int(last),
        cc,
        5,
        0,
        peak,
        lo,
        ret_c,
        entry_price,
    )


_DAY_NS = 86_400_000_000_000


def scan_candidate(
    prices: dict,
    probs: dict,
    sym: str,
    mkt: str,
    t_theta: pd.Timestamp,
    t_e: pd.Timestamp,
    is_earnings: bool,
    p_surge,
    r_surge,
    policy: dict,
):
    """Run the JIT kernel for one candidate.

    Returns ``None`` when no trade is produced, otherwise a tuple
    ``(entry_ts, entry_prob, entry_price, exit_ts, exit_price, reason_code,
    hard_floor_pct, peak, trough, ret_c)`` where ``entry_ts``/``exit_ts`` are the
    original tz-aware Timestamps (so the caller formats dates identically).
    """
    bar_value, bar_norm, bar_high, bar_low, bar_close, bars = _symbol_arrays(prices, sym)
    if bar_value.shape[0] < 2:
        return None
    pt_value, pval_raw, day_uni, pval_uni, points = _market_arrays(probs, mkt)
    if pt_value.shape[0] == 0:
        return None

    window_lo_value = np.int64(t_theta.value) - 30 * _DAY_NS
    t_e_value = np.int64(t_e.value)
    first_eligible_value = np.int64(t_theta.normalize().value)
    resolution_cut_value = np.int64((t_e - pd.Timedelta(days=1)).value)

    surge = float(p_surge) if p_surge is not None else float("nan")
    runup = float(r_surge) if r_surge is not None else float("nan")

    result = _scan(
        bar_value,
        bar_norm,
        bar_high,
        bar_low,
        bar_close,
        pt_value,
        pval_raw,
        day_uni,
        pval_uni,
        window_lo_value,
        t_e_value,
        first_eligible_value,
        resolution_cut_value,
        1 if is_earnings else 0,
        float(policy["enter_strong"]),
        float(policy["enter_floor"]),
        int(policy["hold_days"]),
        float(policy["atr_mult"]),
        float(policy["lock_activate"]),
        float(policy["theta_out"]),
        surge,
        float(policy.get("max_prob_surge", 999.0)),
        runup,
        float(policy.get("max_price_runup", 999.0)),
    )

    if result[0] == 0:
        return None

    (
        _status,
        entry_pt_index,
        _entry_gindex,
        exit_gindex,
        exit_price,
        reason_code,
        hard_floor_pct,
        peak,
        trough,
        ret_c,
        entry_price,
    ) = result

    entry_point = points[int(entry_pt_index)]
    exit_bar = bars[int(exit_gindex)]
    return (
        bars[int(_entry_gindex)][0], # entry_ts (rolled to asset's valid trading day)
        float(entry_point[1]),   # entry_prob
        float(entry_price),
        exit_bar[0],             # exit_ts (original Timestamp)
        float(exit_price),
        int(reason_code),
        int(hard_floor_pct),
        float(peak),
        float(trough),
        float(ret_c),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pure-Python reference + dispatch (formerly pipeline/strategy), long-only
# ══════════════════════════════════════════════════════════════════════════════


def calc_atr(bars: list[tuple]) -> float:
    """Average True Range from (t, h, l, c) bars."""
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h, l, c = bars[i][1], bars[i][2], bars[i][3]
        pc = bars[i - 1][3]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs) / len(trs)


def entry_day(
    prob_path: list[tuple],
    t_theta: pd.Timestamp,
    policy: dict,
) -> tuple[pd.Timestamp, float] | None:
    """Apply the entry rule; return (entry_ts, entry_prob) or None."""
    first_eligible_day = t_theta.normalize()
    pts = [(t, v) for t, v in prob_path if t >= first_eligible_day]
    if not pts:
        return None
    held = 0
    for t, v in pts:
        if v >= policy["enter_strong"]:
            return t, v
        elif v >= policy["enter_floor"]:
            held += 1
            if held >= policy["hold_days"]:
                return t, v
        else:
            held = 0
    return None



def clear_kernel_caches() -> None:
    """Drop cached numpy views AND the polarity-corrected probability paths."""
    clear_caches()
    clear_effective_probs_cache()


def _simulate_one_py(
    row: dict | pd.Series,
    prices: dict[str, list[tuple]],
    probs: dict[str, list[tuple]],
    policy: dict,
    *,
    apply_polarity: bool = True,
) -> dict | None:
    """Reference pure-Python trade simulation. Returns trade dict or None.

    This is the authoritative definition of the trade semantics. The numba
    kernel reproduces it exactly; ``simulate_one`` dispatches to whichever is
    active.
    """
    sym, mkt = row["symbol"], row["market_id"]
    t_theta = pd.Timestamp(row["t_theta"]).tz_convert("UTC")
    t_e = pd.Timestamp(row["t_e"]).tz_convert("UTC")

    question = str(row.get("question", ""))
    if apply_polarity:
        # Re-polarize before anything reads the path: entry_day, theta_out and
        # `converged` must all see "high == bullish". Polarity 0 means neither
        # side is a clean signal for this symbol, so the pair is not tradable.
        polarity, polarity_source = resolve_polarity(question, sym)
        if polarity == 0:
            return None
        probs = effective_probs(probs, mkt, polarity)
    else:
        # Explicit compatibility mode for the live plane. Keeping this opt-out
        # here prevents the research default from silently changing live exits.
        polarity, polarity_source = 1, "raw"

    is_earnings = "earnings" in str(row.get("feat_archetype", "")).lower()
    closes = prices.get(sym, [])

    win = [(t, h, l, c) for t, h, l, c in closes
           if t_theta - pd.Timedelta(days=30) <= t <= t_e]
    if len(win) < 2:
        return None

    ent = entry_day(probs.get(mkt, []), t_theta, policy)
    if ent is None:
        return None
    entry_ts = ent[0]

    p_surge = effective_prob_surge(row, polarity)
    r_surge = row.get("feat_runup_since_t0")
    if p_surge is not None and p_surge > policy.get("max_prob_surge", 999.0):
        return None
    if r_surge is not None and r_surge > policy.get("max_price_runup", 999.0):
        return None

    entry_idx = next((i for i, b in enumerate(win) if b[0] >= entry_ts), -1)
    if entry_idx == -1:
        return None
    resolution_cut = t_e - pd.Timedelta(days=1)
    if win[entry_idx][0] >= resolution_cut:
        return None

    path = [bar for bar in win[entry_idx:] if bar[0] < resolution_cut]
    if len(path) < 2:
        return None

    hist_bars = win[max(0, entry_idx - 15):entry_idx + 1]
    atr = calc_atr(hist_bars)

    entry_price = path[0][3]
    if atr == 0 or entry_price == 0:
        return None
    atr_pct = atr / entry_price

    prob_path = {t.normalize(): v for t, v in probs.get(mkt, [])}

    atr_mult = policy["atr_mult"]
    lock_activate = policy["lock_activate"]
    theta_out = policy["theta_out"]
    peak = 0.0

    for i, (t, h, l, c) in enumerate(path):
        ret_c = c / entry_price - 1.0
        ret_h = h / entry_price - 1.0
        ret_l = l / entry_price - 1.0

        reason = None
        if i > 0:
            stop_dist = atr_mult * atr_pct

            if prob_path.get(t.normalize(), 1.0) < theta_out:
                reason = f"poly<{theta_out}"
            elif ret_l <= peak - stop_dist:
                reason = f"trailing_{atr_mult:.1f}ATR"
                c = max(l, entry_price * (1.0 + peak - stop_dist))
                ret_c = c / entry_price - 1.0
            elif peak >= lock_activate:
                hard_floor_pct = int(peak * 100)
                hard_floor = hard_floor_pct / 100.0
                if ret_l < hard_floor:
                    reason = f"profit_lock_{hard_floor_pct}%"
                    c = max(l, entry_price * (1.0 + hard_floor))
                    ret_c = c / entry_price - 1.0

            if reason is None and i == len(path) - 1:
                reason = "resolution-1d"

        if reason:
            lo = min(ll / entry_price - 1.0 for _, _, ll, _ in path[:i + 1])
            # `probs` is the polarity-corrected view, so `converged` reports
            # whether the bullish thesis resolved true -- not raw YES.
            mkt_probs = probs.get(mkt, [])
            converged = "YES" if mkt_probs and mkt_probs[-1][1] >= 0.5 else "NO" if mkt_probs else "UNKNOWN"
            return dict(
                market_id=mkt, symbol=sym,
                question=question,
                polarity=polarity,
                polarity_source=polarity_source,
                pct=round(ent[1], 3),
                converged=converged,
                asset_confidence=row.get("confidence_score"),
                question_confidence=row.get("feat_llm_confidence"),
                archetype=row.get("feat_archetype", ""),
                relevance=round(float(row.get(RELEVANCE_COL, 0)), 3),
                split=row.get("split", ""),
                entry_date=str(path[0][0].date()), entry_prob=round(ent[1], 3),
                entry_price=round(entry_price, 2),
                exit_date=str(t.date()), exit_price=round(c, 2),
                exit_reason=reason,
                peak_pct=round(peak * 100, 2), trough_pct=round(lo * 100, 2),
                return_pct=round(ret_c * 100, 2),
            )

        if i == 0:
            peak = 0.0
        else:
            peak = max(peak, ret_h)

    t, h, l, c = path[-1]
    ret_c = c / entry_price - 1.0
    lo = min(ll / entry_price - 1.0 for _, _, ll, _ in path)
    mkt_probs = probs.get(mkt, [])
    converged = "YES" if mkt_probs and mkt_probs[-1][1] >= 0.5 else "NO" if mkt_probs else "UNKNOWN"
    return dict(
        market_id=mkt, symbol=sym,
        question=question,
        polarity=polarity,
        polarity_source=polarity_source,
        pct=round(ent[1], 3),
        converged=converged,
        asset_confidence=row.get("confidence_score"),
        question_confidence=row.get("feat_llm_confidence"),
        archetype=row.get("feat_archetype", ""),
        relevance=round(float(row.get(RELEVANCE_COL, 0)), 3),
        split=row.get("split", ""),
        entry_date=str(path[0][0].date()), entry_prob=round(ent[1], 3),
        entry_price=round(entry_price, 2),
        exit_date=str(t.date()), exit_price=round(c, 2),
        exit_reason="end_of_window",
        peak_pct=round(peak * 100, 2), trough_pct=round(lo * 100, 2),
        return_pct=round(ret_c * 100, 2),
    )


def simulate_one(
    row: dict | pd.Series,
    prices: dict[str, list[tuple]],
    probs: dict[str, list[tuple]],
    policy: dict,
) -> dict | None:
    """Simulate a single trade. Returns trade result dict or None.

    Dispatches to the numba kernel (``scan_candidate``) when available
    (``_USE_KERNEL``), otherwise to the pure-Python reference
    ``_simulate_one_py``. Both return identical dicts; the kernel only removes
    the per-call pandas/list overhead so the CEM search runs faster. The
    timestamp and string formatting below stays in Python so the fast path is
    timezone-correct and byte-for-byte compatible with the reference.
    """
    if not _USE_KERNEL:
        return _simulate_one_py(row, prices, probs, policy)

    sym, mkt = row["symbol"], row["market_id"]

    # The kernel never sees `question`, so polarity must be resolved here. See
    # `core.polarity` for why this is a flip (or a skip), never a short.
    question = str(row.get("question", ""))
    polarity, polarity_source = resolve_polarity(question, sym)
    if polarity == 0:
        return None
    probs = effective_probs(probs, mkt, polarity)

    t_theta = pd.Timestamp(row["t_theta"]).tz_convert("UTC")
    t_e = pd.Timestamp(row["t_e"]).tz_convert("UTC")
    is_earnings = "earnings" in str(row.get("feat_archetype", "")).lower()

    scanned = scan_candidate(
        prices,
        probs,
        sym,
        mkt,
        t_theta,
        t_e,
        is_earnings,
        effective_prob_surge(row, polarity),
        row.get("feat_runup_since_t0"),
        policy,
    )
    if scanned is None:
        return None

    (
        entry_ts,
        entry_prob,
        entry_price,
        exit_ts,
        exit_price,
        reason_code,
        hard_floor_pct,
        peak,
        trough,
        ret_c,
    ) = scanned

    if reason_code == 1:
        reason = f"trailing_{policy['atr_mult']:.1f}ATR"
    elif reason_code == 2:
        reason = f"profit_lock_{hard_floor_pct}%"
    elif reason_code == 3:
        reason = f"poly<{policy['theta_out']}"
    elif reason_code == 4:
        reason = "resolution-1d"
    else:
        reason = "end_of_window"

    # `probs` is the polarity-corrected view, so `converged` reports whether the
    # bullish thesis resolved true -- not whether the raw market resolved YES.
    mkt_probs = probs.get(mkt, [])
    converged = "YES" if mkt_probs and mkt_probs[-1][1] >= 0.5 else "NO" if mkt_probs else "UNKNOWN"
    return dict(
        market_id=mkt, symbol=sym,
        question=question,
        polarity=polarity,
        polarity_source=polarity_source,
        pct=round(entry_prob, 3),
        converged=converged,
        asset_confidence=row.get("confidence_score"),
        question_confidence=row.get("feat_llm_confidence"),
        archetype=row.get("feat_archetype", ""),
        relevance=round(float(row.get(RELEVANCE_COL, 0)), 3),
        split=row.get("split", ""),
        entry_date=str(entry_ts.date()), entry_prob=round(entry_prob, 3),
        entry_price=round(entry_price, 2),
        exit_date=str(exit_ts.date()), exit_price=round(exit_price, 2),
        exit_reason=reason,
        peak_pct=round(peak * 100, 2), trough_pct=round(trough * 100, 2),
        return_pct=round(ret_c * 100, 2),
    )
