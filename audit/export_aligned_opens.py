"""Align downloaded Yahoo opens to the exact H/L/C scale in data/prices.pkl."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_artifacts"


def main() -> None:
    with open(OUT / "yahoo_daily_ohlc.pkl", "rb") as handle:
        yahoo = pickle.load(handle)
    with open(ROOT / "data" / "prices.pkl", "rb") as handle:
        prices = pickle.load(handle)

    rows: list[dict] = []
    symbol_rows: list[dict] = []
    opens: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    total_bars = 0
    total_aligned = 0

    for symbol, bars in prices.items():
        frame = yahoo.get(symbol)
        aligned: list[tuple[pd.Timestamp, float]] = []
        scales: list[float] = []
        for ts, high, low, close in bars:
            total_bars += 1
            day = pd.Timestamp(ts)
            day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
            day = day.normalize()
            if frame is None or day not in frame.index:
                continue
            downloaded = frame.loc[day]
            if isinstance(downloaded, pd.DataFrame):
                downloaded = downloaded.iloc[-1]
            raw_open = downloaded.get("Open")
            raw_close = downloaded.get("Close")
            if pd.isna(raw_open) or pd.isna(raw_close) or float(raw_close) == 0.0:
                continue
            # This date-specific factor puts Open on the exact scale of the frozen
            # H/L/C artifact and handles later splits or Yahoo repair adjustments.
            scale = float(close) / float(raw_close)
            open_aligned = float(raw_open) * scale
            aligned.append((day, open_aligned))
            scales.append(scale)
            rows.append(
                {
                    "symbol": symbol,
                    "date": str(day.date()),
                    "open": open_aligned,
                    "raw_yahoo_open": float(raw_open),
                    "daily_scale_factor": scale,
                    "frozen_close": float(close),
                    "raw_yahoo_close": float(raw_close),
                }
            )
            total_aligned += 1
        opens[symbol] = aligned
        symbol_rows.append(
            {
                "symbol": symbol,
                "bars": len(bars),
                "aligned_opens": len(aligned),
                "coverage": len(aligned) / max(len(bars), 1),
                "median_scale_factor": float(np.median(scales)) if scales else np.nan,
                "min_scale_factor": float(np.min(scales)) if scales else np.nan,
                "max_scale_factor": float(np.max(scales)) if scales else np.nan,
            }
        )

    with open(OUT / "aligned_opens.pkl", "wb") as handle:
        pickle.dump(opens, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(OUT / "aligned_opens.csv.gz", index=False, compression="gzip")
    by_symbol = pd.DataFrame(symbol_rows).sort_values(["coverage", "symbol"])
    by_symbol.to_csv(OUT / "aligned_open_coverage.csv", index=False)

    summary = {
        "symbols": len(prices),
        "total_frozen_bars": total_bars,
        "total_aligned_opens": total_aligned,
        "bar_coverage": total_aligned / max(total_bars, 1),
        "symbols_below_95pct_coverage": by_symbol.loc[by_symbol.coverage < 0.95, "symbol"].tolist(),
        "symbols_with_no_aligned_open": by_symbol.loc[by_symbol.aligned_opens == 0, "symbol"].tolist(),
        "scale_rule": "aligned_open = raw_yahoo_open * frozen_close / raw_yahoo_close for the same date",
    }
    (OUT / "aligned_open_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["bar_coverage"] < 0.97:
        raise SystemExit(f"Aligned open coverage below 97%: {summary['bar_coverage']:.3%}")


if __name__ == "__main__":
    main()
