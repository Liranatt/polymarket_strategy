"""Run the existing CEM matrix with the isolated gap-aware OHLC kernel."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from audit import ohlc_kernel

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiments", nargs="+", default=["all"])
    parser.add_argument("--benchmarks", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--cem-iters", type=int, default=6)
    parser.add_argument("--cem-pop", type=int, default=20)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    with open(ROOT / "data" / "opens.pkl", "rb") as f:
        opens = pickle.load(f)
    ohlc_kernel.set_open_prices(opens)

    import backtesting.optimize_cem as cem
    cem.simulate_one = ohlc_kernel.simulate_one
    cem.clear_kernel_caches = ohlc_kernel.clear_kernel_caches

    argv = [
        "--experiments", *args.experiments,
        "--benchmarks", *args.benchmarks,
        "--seed", str(args.seed),
        "--cem-iters", str(args.cem_iters),
        "--cem-pop", str(args.cem_pop),
        "--run-id", args.run_id,
        "--candidates-path", "data/candidates_audit_clean.parquet",
    ]
    if not args.detailed:
        argv += ["--no-allocation-log", "--no-forensics-log"]
    cem.main(argv)


if __name__ == "__main__":
    main()
