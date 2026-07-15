"""Fetch raw Yahoo daily OHLC concurrently and validate it against the frozen price manifest."""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = ROOT / "audit_artifacts"
OUT.mkdir(exist_ok=True)


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def unix_day(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def fetch_one(symbol: str, start: str, end: str) -> tuple[str, pd.DataFrame | None, str]:
    target = yahoo_symbol(symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{target}"
    params = {
        "period1": unix_day(start),
        "period2": unix_day(end) + 86400,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 quantitative-research-audit"}
    last_error = ""
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(5, 20))
            if response.status_code == 429:
                last_error = "HTTP 429"
                time.sleep(min(20.0, (2 ** attempt) + random.random()))
                continue
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise RuntimeError(str(chart["error"]))
            results = chart.get("result") or []
            if not results:
                raise RuntimeError("empty chart result")
            result = results[0]
            timestamps = result.get("timestamp") or []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            adj_block = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
            if not timestamps:
                raise RuntimeError("no timestamps")
            frame = pd.DataFrame(
                {
                    "Open": quote.get("open", []),
                    "High": quote.get("high", []),
                    "Low": quote.get("low", []),
                    "Close": quote.get("close", []),
                    "Adj Close": adj_block.get("adjclose", [math.nan] * len(timestamps)),
                    "Volume": quote.get("volume", []),
                },
                index=pd.to_datetime(timestamps, unit="s", utc=True).normalize(),
            )
            frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna(how="all")
            if frame.empty or "Open" not in frame or frame["Open"].notna().sum() == 0:
                raise RuntimeError("missing open")
            return symbol, frame, ""
        except Exception as exc:  # network/data failures are recorded, never hidden
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(12.0, (1.5 ** attempt) + random.random()))
    return symbol, None, last_error


def rel_error(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale


def main() -> None:
    manifest = json.loads((AUDIT / "frozen_symbol_manifest.json").read_text(encoding="utf-8"))
    symbols = list(manifest["symbols"])
    start, end = manifest["start"], manifest["end"]

    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_one, s, start, end): s for s in symbols}
        for number, future in enumerate(cf.as_completed(futures), start=1):
            symbol, frame, error = future.result()
            if frame is None:
                failures[symbol] = error
            else:
                frames[symbol] = frame
            if number % 50 == 0 or number == len(symbols):
                print(f"completed={number}/{len(symbols)} success={len(frames)} failed={len(failures)}", flush=True)

    # The uploaded artifact contains (timestamp, high, low, close). Validate scale/date alignment.
    with open(ROOT / "data" / "prices.pkl", "rb") as handle:
        frozen_prices = pickle.load(handle)

    rows: list[dict[str, Any]] = []
    all_errors: list[float] = []
    symbols_with_overlap = 0
    for symbol, bars in frozen_prices.items():
        fetched = frames.get(symbol)
        if fetched is None or fetched.empty:
            continue
        local = {
            pd.Timestamp(ts).tz_convert("UTC").normalize(): (float(high), float(low), float(close))
            for ts, high, low, close in bars
        }
        overlap = sorted(set(local).intersection(fetched.index))
        errors: list[float] = []
        for date in overlap:
            high, low, close = local[date]
            row = fetched.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            for original, downloaded in ((high, row["High"]), (low, row["Low"]), (close, row["Close"])):
                if pd.notna(downloaded):
                    errors.append(rel_error(original, float(downloaded)))
        if overlap:
            symbols_with_overlap += 1
            all_errors.extend(errors)
        rows.append(
            {
                "symbol": symbol,
                "frozen_rows": len(bars),
                "downloaded_rows": 0 if fetched is None else len(fetched),
                "overlap_days": len(overlap),
                "median_relative_hlc_error": float(pd.Series(errors).median()) if errors else math.nan,
                "p99_relative_hlc_error": float(pd.Series(errors).quantile(0.99)) if errors else math.nan,
            }
        )

    coverage = len(frames) / max(len(symbols), 1)
    overall_median = float(pd.Series(all_errors).median()) if all_errors else math.inf
    overall_p99 = float(pd.Series(all_errors).quantile(0.99)) if all_errors else math.inf
    alignment = pd.DataFrame(rows).sort_values("symbol")
    alignment.to_csv(OUT / "yahoo_ohlc_alignment.csv", index=False)

    with open(OUT / "yahoo_daily_ohlc.pkl", "wb") as handle:
        pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "manifest_prices_sha256": manifest["source_prices_sha256"],
        "requested_symbols": len(symbols),
        "downloaded_symbols": len(frames),
        "symbol_coverage": coverage,
        "failed_symbols": failures,
        "symbols_with_hlc_overlap": symbols_with_overlap,
        "overall_median_relative_hlc_error": overall_median,
        "overall_p99_relative_hlc_error": overall_p99,
        "start": start,
        "end": end,
        "total_rows": int(sum(len(frame) for frame in frames.values())),
    }
    (OUT / "yahoo_daily_ohlc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if coverage < 0.95:
        raise SystemExit(f"Yahoo OHLC symbol coverage below 95%: {coverage:.3%}")
    if symbols_with_overlap / max(len(symbols), 1) < 0.95:
        raise SystemExit("HLC overlap coverage below 95%")
    if overall_median > 1e-6:
        raise SystemExit(f"Raw Yahoo HLC scale mismatch: median relative error={overall_median:.6g}")
    if overall_p99 > 0.01:
        raise SystemExit(f"Raw Yahoo HLC material tail mismatch: p99 relative error={overall_p99:.6g}")


if __name__ == "__main__":
    main()
