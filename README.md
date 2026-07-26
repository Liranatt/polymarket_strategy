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

## Requirements

Python **3.11+** (developed and verified on 3.13). Install from the repository
root:

```bash
pip install -r requirements.txt
```

`numba` is optional but **strongly recommended** — it JIT-compiles the trade
kernel. Without it the run still produces identical results, but the CEM search
falls back to a much slower pure-Python path.

## Quick start

Run everything from the **repository root**. The driver reads `data/prices.pkl`
and `data/probs.pkl` as paths relative to the working directory, so it must be
launched from there. (The notebook is more forgiving: it walks up from its own
location to find the repo root and `chdir`s there itself.)

**Option A — the notebook** (documentation + mapping table + backtest + figures):

```bash
jupyter notebook notebooks/standard_cem_strategy.ipynb
```

Then *Run All*. The notebook is committed **fully executed**, so every result,
table and figure can be read without running anything.

**Option B — the backtest directly.** The notebook does not reimplement the
strategy; its backtest cell shells out to exactly this command, so running it
yourself reproduces the same numbers:

```bash
python -m backtesting.optimize_cem --experiments baseline --benchmarks SPY QQQ --seed 42 --run-id my_run
```

Takes about **1 minute** with `numba` installed. Everything runs **offline** —
no Polymarket API, no Gemini calls, no database. The only inputs are the four
committed artifacts under `data/`.

## Where the output goes

| Path | Written by | Committed? |
|---|---|---|
| `notebooks/outputs/` | the notebook — mapping CSV, per-benchmark trade and daily-equity CSVs, summary CSV, fitted CEM parameters, both comparison figures | yes |
| `runs/<run-id>/` | the driver, when `--run-id` is passed — full audit trail: results CSV, train/test trade and equity logs, allocation and disposition logs, trade forensics, the complete CEM fitness population | no (gitignored) |

The notebook uses `--run-id standard_cem_notebook`. **Pass your own `--run-id`**
when driving the CLI directly: if you omit it the driver falls back to the
historical un-namespaced layout and scatters result CSVs into `data/` and
`output/` alongside the committed input artifacts.

## Useful flags

```
--experiments baseline        the simple standard CEM arm (this repo's subject)
--benchmarks SPY QQQ          one or both benchmarks
--seed 42                     CEM base seed; QQQ adds a fixed +10000 offset
--cem-iters 6 --cem-pop 20    search budget — lower these for a fast smoke test
--run-id NAME                 namespace outputs under runs/NAME/
--no-allocation-log           skip per-candidate decision logging (faster)
--no-forensics-log            skip the derived trade-forensics CSV (faster)
--candidates-path PATH        defaults to the audit-clean artifact below
```

`--candidates-path` defaults to `data/candidates_audit_clean.parquet`, which is
the only candidate artifact shipped here and the one behind every number in this
repository. Its help text mentions a legacy `data/candidates.parquet` for
explicit comparison runs; that file belongs to the full research repository and
is deliberately **not** included.

`python -m backtesting.optimize_cem --help` lists everything.

## What is in here

| Path | Contents |
|---|---|
| `notebooks/standard_cem_strategy.ipynb` | requirements, data-pipeline documentation, the complete question → stock/ETF mapping, the standard CEM backtest on SPY and QQQ, equity-curve figures, verification |
| `notebooks/outputs/` | the practical outputs (see table above) |
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

The backtest is deterministic for a given seed. The notebook's final cell
asserts the run against the canonical standard-CEM results and fails loudly on
any drift. Running the Quick-start command above reproduces, exactly:

| Benchmark | Portfolio | Buy & hold | Excess | Sharpe | Max DD | Trades |
|---|---|---|---|---|---|---|
| SPY | +30.2742% | +8.5199% | +21.7543% | 3.2975 | −6.6651% | 225 |
| QQQ | +27.3641% | +17.5912% | +9.7729% | 2.9321 | −6.3585% | 223 |

Verified on Python 3.13.3 with numpy 2.4.4, pandas 3.0.2, pyarrow 24.0.0 and
numba 0.65.1. Out-of-sample window: 2026-01-02 → 2026-06-12 (OOS boundary
2026-01-01); initial capital $100,000; IB-style transaction costs on all four
rotation legs (benchmark sell → asset buy → asset sell → benchmark re-buy).

*Research code. Nothing here is investment advice.*
