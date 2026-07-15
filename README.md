# Polymarket Strategy — the Simple Standard CEM Backtest

A rule-based event strategy that trades a lag in information diffusion: when a
**Polymarket** prediction market re-prices sharply on a catalyst, the
fundamentally-exposed **US equity** often re-prices more slowly. Polymarket's
probability is used as an upstream *oracle* — the trade is on the delayed
equity move, not on Polymarket itself. Entry/exit rules are optimized with the
**Cross-Entropy Method (CEM)** on a leakage-safe chronological split.

This repository is a self-contained cut of the research project: the shared
trade kernel, the CEM backtest driver, the committed data artifacts, and one
notebook that presents and runs the **simple standard CEM strategy** (the
`Baseline` arm — no treatment ladder, no walk-forward refits, no Kelly sizing,
no PPO) against **SPY** and **QQQ**.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/standard_cem_strategy.ipynb   # Run All
```

The notebook runs **offline** — no Polymarket API, no Gemini calls, no
database. Its only inputs are the four committed artifacts under `data/`.
It is committed fully executed, so results can be read without running it.

## What is in here

| Path | Contents |
|---|---|
| `notebooks/standard_cem_strategy.ipynb` | requirements, data-pipeline documentation, the complete question → stock/ETF mapping, the standard CEM backtest on SPY and QQQ, equity-curve figures, verification |
| `notebooks/outputs/` | the practical outputs: mapping CSV, per-benchmark trade and daily-equity CSVs, summary CSV, fitted CEM parameters, both comparison figures |
| `core/` | the shared entry/exit trade kernel, policy parameter space, polarity resolution |
| `backtesting/` | the CEM search + portfolio-simulation driver (`optimize_cem.py`) |
| `data/candidates_audit_clean.parquet` | audit-cleaned candidate set: one row per Polymarket-market × symbol, with point-in-time features and the chronological split |
| `data/prices.pkl` | `{symbol: [(ts, high, low, close), …]}` daily bars, incl. SPY/QQQ |
| `data/probs.pkl` | `{market_id: [(ts, P(YES)), …]}` daily probability paths |
| `data/polarity_labels.json` | committed per-(question, symbol) signal-polarity labels |

The upstream ingestion pipeline (Polymarket Gamma scan → deterministic
filtering → bracket-ladder dedup → Gemini relevance scoring and asset mapping
→ polarity labeling → artifact build) is **documented in the notebook** but
not included — the committed artifacts are its frozen output.

## Reproducibility

The backtest is deterministic for a given seed. The notebook runs the driver
with the project defaults (`seed 42`, 6 CEM iterations × population 20) and
verifies the resulting SPY/QQQ metrics against the canonical standard-CEM
experiment. The same run can be reproduced from the command line:

```bash
python -m backtesting.optimize_cem --experiments baseline --benchmarks SPY QQQ --seed 42 --run-id standard_cem_notebook
```

Out-of-sample window: 2026-01-02 → 2026-06-12 (OOS boundary 2026-01-01);
initial capital $100,000; IB-style transaction costs on all four rotation legs.

*Research code. Nothing here is investment advice.*
