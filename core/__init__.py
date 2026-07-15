"""Shared, plane-agnostic core: policy, features, and the entry/exit kernel.

Pure — no database, no network. Imported by BOTH the backtest
(`backtesting.optimize_cem`, via the numba fast path) and the live trader
(`live`, via the pure-Python reference), so the two planes can never trade
different rules. Populated during the core-extraction step.
"""
