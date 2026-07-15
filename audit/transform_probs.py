from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["delay1d", "circular_placebo", "restore"])
    parser.add_argument("--path", type=Path, default=Path("data/probs.pkl"))
    parser.add_argument("--backup", type=Path, default=Path("data/probs.audit_original.pkl"))
    args = parser.parse_args()

    if args.mode == "restore":
        if not args.backup.exists():
            raise FileNotFoundError(args.backup)
        args.path.write_bytes(args.backup.read_bytes())
        return

    if not args.backup.exists():
        args.backup.write_bytes(args.path.read_bytes())

    with args.backup.open("rb") as handle:
        probs = pickle.load(handle)

    transformed: dict = {}
    rng = np.random.default_rng(20260715)
    for market_id, points in probs.items():
        normalized = [(pd.Timestamp(ts), float(value)) for ts, value in points]
        if args.mode == "delay1d":
            transformed[market_id] = [(ts + pd.Timedelta(days=1), value) for ts, value in normalized]
        else:
            if len(normalized) < 3:
                transformed[market_id] = normalized
                continue
            dates = [item[0] for item in normalized]
            values = np.asarray([item[1] for item in normalized], dtype=float)
            shift = int(rng.integers(1, len(values)))
            shifted = np.roll(values, shift)
            transformed[market_id] = list(zip(dates, shifted.tolist()))

    with args.path.open("wb") as handle:
        pickle.dump(transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
