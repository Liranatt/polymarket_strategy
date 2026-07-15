"""Deterministic execution tests for daily OHLC ordering and gap fills."""
from __future__ import annotations

import pandas as pd

from audit.ohlc_kernel import set_open_prices, simulate_one


def ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def make_row():
    return pd.Series({
        "symbol": "XYZ", "market_id": "M1", "question": "Will XYZ earnings beat?",
        "t_theta": ts("2026-01-03"), "t_e": ts("2026-01-08"),
        "feat_archetype": "earnings", "feat_prob_surge_since_t0": 0.1,
        "feat_runup_since_t0": 0.0, "feat_connection_strength": 1.0,
        "split": "test",
    })


POLICY = {
    "enter_strong": 0.60, "enter_floor": 0.55, "hold_days": 1,
    "atr_mult": 1.0, "lock_activate": 0.05, "theta_out": 0.45,
    "max_prob_surge": 1.0, "max_price_runup": 1.0,
}


def run_case(day4_open: float, day4_low: float):
    prices = {"XYZ": [
        (ts("2026-01-01"), 101.0, 99.0, 100.0),
        (ts("2026-01-02"), 101.0, 99.0, 100.0),
        (ts("2026-01-03"), 101.0, 99.0, 100.0),
        (ts("2026-01-04"), 120.0, 99.5, 115.0),
        (ts("2026-01-05"), 116.0, day4_low, 110.0),
        (ts("2026-01-06"), 111.0, 105.0, 108.0),
    ]}
    set_open_prices({"XYZ": [
        (ts("2026-01-01"), 100.0), (ts("2026-01-02"), 100.0),
        (ts("2026-01-03"), 100.0), (ts("2026-01-04"), 100.0),
        (ts("2026-01-05"), day4_open), (ts("2026-01-06"), 108.0),
    ]})
    probs = {"M1": [(ts("2026-01-03"), 0.70), (ts("2026-01-04"), 0.70),
                     (ts("2026-01-05"), 0.70), (ts("2026-01-06"), 0.70)]}
    return simulate_one(make_row(), prices, probs, POLICY)


def main():
    gap = run_case(115.0, 105.0)
    assert gap is not None and gap["exit_price"] == 115.0 and gap["stop_gap_fill"]
    through = run_case(121.0, 105.0)
    assert through is not None and through["exit_price"] == 120.0 and not through["stop_gap_fill"]
    assert gap["exit_date"] == "2026-01-05"
    print("OHLC execution invariants passed")


if __name__ == "__main__":
    main()
