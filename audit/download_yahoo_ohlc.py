"""Download Yahoo daily OHLC for the exact original symbol universe.

This file intentionally does not run the backtest. It only creates an external
Open-price supplement. The local audit later joins these rows to the frozen
original (High, Low, Close) bars and rejects scale/date mismatches.
"""
from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SYMBOL_FILE = ROOT / "audit" / "original_symbols.txt"
OUT_DIR = ROOT / "audit" / "yahoo_output"
START = "2021-12-20"
END_EXCLUSIVE = "2026-07-06"
MAX_WORKERS = 6
RETRIES = 4


def yahoo_symbol(symbol: str) -> str:
    # Yahoo represents US class shares with a dash. This is harmless for the
    # current list and keeps the downloader correct if a dotted symbol appears.
    return symbol.replace(".", "-")


def flatten_single_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    levels0 = set(map(str, frame.columns.get_level_values(0)))
    levels1 = set(map(str, frame.columns.get_level_values(1)))
    if ticker in levels1:
        return frame.xs(ticker, axis=1, level=1, drop_level=True)
    if ticker in levels0:
        return frame.xs(ticker, axis=1, level=0, drop_level=True)
    # A one-ticker response can still have a redundant unnamed level.
    if len(levels0) == 1:
        return frame.droplevel(0, axis=1)
    if len(levels1) == 1:
        return frame.droplevel(1, axis=1)
    raise ValueError(f"cannot identify ticker level for {ticker}: {frame.columns}")


def download_one(symbol: str) -> tuple[pd.DataFrame | None, dict]:
    ticker = yahoo_symbol(symbol)
    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            frame = yf.download(
                ticker,
                start=START,
                end=END_EXCLUSIVE,
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                progress=False,
                threads=False,
                timeout=40,
            )
            frame = flatten_single_ticker(frame, ticker)
            if frame.empty:
                raise ValueError("empty Yahoo response")
            required = {"Open", "High", "Low", "Close"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")

            out = pd.DataFrame(index=frame.index)
            out["symbol"] = symbol
            out["yahoo_symbol"] = ticker
            out["date"] = pd.to_datetime(frame.index).date.astype(str)
            for source, target in [
                ("Open", "open_raw"),
                ("High", "high_raw"),
                ("Low", "low_raw"),
                ("Close", "close_raw"),
                ("Adj Close", "adj_close"),
                ("Volume", "volume"),
                ("Dividends", "dividends"),
                ("Stock Splits", "stock_splits"),
            ]:
                if source in frame.columns:
                    out[target] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
                else:
                    out[target] = np.nan

            factor = out["adj_close"] / out["close_raw"]
            factor = factor.where(np.isfinite(factor) & (factor > 0), 1.0)
            out["adj_factor"] = factor
            out["open_adjusted"] = out["open_raw"] * factor
            out["high_adjusted"] = out["high_raw"] * factor
            out["low_adjusted"] = out["low_raw"] * factor
            out["close_adjusted"] = out["close_raw"] * factor
            out = out.reset_index(drop=True)
            out = out.dropna(subset=["open_raw", "high_raw", "low_raw", "close_raw"])
            return out, {
                "symbol": symbol,
                "yahoo_symbol": ticker,
                "status": "ok",
                "rows": int(len(out)),
                "first_date": out["date"].min() if len(out) else "",
                "last_date": out["date"].max() if len(out) else "",
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:  # Yahoo intermittently rate-limits individual calls.
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(3 * attempt, 10))
    return None, {
        "symbol": symbol,
        "yahoo_symbol": ticker,
        "status": "failed",
        "rows": 0,
        "first_date": "",
        "last_date": "",
        "attempts": RETRIES,
        "error": last_error,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [line.strip() for line in SYMBOL_FILE.read_text().splitlines() if line.strip()]
    frames: list[pd.DataFrame] = []
    logs: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, symbol): symbol for symbol in symbols}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            frame, log = future.result()
            logs.append(log)
            if frame is not None:
                frames.append(frame)
            print(f"[{done:03d}/{len(symbols)}] {log['symbol']}: {log['status']} rows={log['rows']}", flush=True)

    log_df = pd.DataFrame(logs).sort_values("symbol")
    log_df.to_csv(OUT_DIR / "download_log.csv", index=False)
    if not frames:
        raise RuntimeError("Yahoo download produced no rows")

    data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    data.to_parquet(OUT_DIR / "yahoo_daily_ohlc.parquet", index=False)
    data.to_csv(OUT_DIR / "yahoo_daily_ohlc.csv.gz", index=False, compression="gzip")
    summary = {
        "requested_symbols": len(symbols),
        "successful_symbols": int((log_df["status"] == "ok").sum()),
        "failed_symbols": int((log_df["status"] != "ok").sum()),
        "rows": int(len(data)),
        "start": START,
        "end_exclusive": END_EXCLUSIVE,
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "pandas_version": pd.__version__,
    }
    (OUT_DIR / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
