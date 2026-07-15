"""Fetch raw Yahoo daily OHLC using a manifest derived from the uploaded price artifact."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = ROOT / "audit_artifacts"
OUT.mkdir(exist_ok=True)


def ys(symbol: str) -> str:
    return symbol.replace(".", "-")


def download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=" ".join(ys(s) for s in symbols),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=True,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30,
    )


def extract(batch: pd.DataFrame, symbol: str, n: int) -> pd.DataFrame:
    target = ys(symbol)
    if batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns, pd.MultiIndex):
        l0 = set(map(str, batch.columns.get_level_values(0)))
        l1 = set(map(str, batch.columns.get_level_values(1)))
        if target in l0:
            frame = batch[target].copy()
        elif target in l1:
            frame = batch.xs(target, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if n != 1:
            return pd.DataFrame()
        frame = batch.copy()
    frame.columns = [str(c).title() for c in frame.columns]
    frame.index = pd.to_datetime(frame.index, utc=True).normalize()
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in frame]
    return frame[keep].dropna(how="all").sort_index()


def main() -> None:
    manifest = json.loads((AUDIT / "frozen_symbol_manifest.json").read_text())
    symbols = list(manifest["symbols"])
    start, end = manifest["start"], manifest["end"]
    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    chunk_size = 35
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        batch = pd.DataFrame()
        for attempt in range(3):
            try:
                batch = download(chunk, start, end)
                if not batch.empty:
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)
        for symbol in chunk:
            frame = extract(batch, symbol, len(chunk))
            if frame.empty or "Open" not in frame:
                failures.append(symbol)
            else:
                frames[symbol] = frame

    still_failed: list[str] = []
    for symbol in failures:
        frame = pd.DataFrame()
        for attempt in range(3):
            try:
                frame = extract(download([symbol], start, end), symbol, 1)
                if not frame.empty and "Open" in frame:
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)
        if frame.empty or "Open" not in frame:
            still_failed.append(symbol)
        else:
            frames[symbol] = frame

    with open(OUT / "yahoo_daily_ohlc.pkl", "wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        "manifest_prices_sha256": manifest["source_prices_sha256"],
        "requested_symbols": len(symbols),
        "downloaded_symbols": len(frames),
        "failed_symbols": still_failed,
        "start": start,
        "end": end,
        "total_rows": int(sum(len(frame) for frame in frames.values())),
    }
    (OUT / "yahoo_daily_ohlc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(frames) / max(len(symbols), 1) < 0.95:
        raise SystemExit("Yahoo OHLC symbol coverage below 95%")


if __name__ == "__main__":
    main()
