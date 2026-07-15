"""Single source of truth for the trading policy parameter space and sizing.

Both planes import from here so they can never disagree:
  * `backtesting.optimize_cem` — CEM search over these bounds, half-Kelly sizing.
  * `live.policy` / `live.strategy_engine` — trades the fitted policy, same bounds.

Historically `DEFAULT_POLICY`/`CEM_BOUNDS` lived in `pipeline.strategy` and the
Kelly bounds were duplicated (and had drifted: optimize_cem used 0.05/0.20, live
used 0.03/0.15). The live values are unified onto the backtest values here, so
live sizing matches the policy that was actually fitted.
"""
from __future__ import annotations

# Column carrying the Gemini-derived relevance / connection strength.
RELEVANCE_COL = "feat_connection_strength"

# Default policy (also the CEM search's starting mean, via optimize_cem's
# PORT_DEFAULT which layers position sizing on top).
DEFAULT_POLICY = dict(
    atr_mult=3.65,
    lock_activate=0.03,
    theta_out=0.55,
    enter_strong=0.75,
    enter_floor=0.70,
    hold_days=1,
    max_prob_surge=0.40,
    max_price_runup=0.10,
)

# CEM samples policies inside these bounds. `theta_out` lower bound is 0.45 and
# `enter_floor` lower bound is 0.55 — entry is governed purely by these knobs,
# there is no separate hard entry threshold.
CEM_BOUNDS = dict(
    atr_mult=(1.5, 4.0),
    lock_activate=(0.02, 0.10),
    theta_out=(0.45, 0.60),
    enter_strong=(0.60, 0.85),
    enter_floor=(0.55, 0.80),
    hold_days=(1, 5),
    max_prob_surge=(0.20, 0.80),
    max_price_runup=(0.02, 0.20),
)

# Half-Kelly sizing bounds — the backtest values, now shared with live.
KELLY_MIN_N = 10
KELLY_LOOKBACK_N = 30
KELLY_MIN_SZ = 0.05
KELLY_MAX_SZ = 0.20
