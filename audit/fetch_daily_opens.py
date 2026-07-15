"""Fetch Yahoo daily opens and align them to the frozen H/L/C artifact."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "audit_artifacts"
OUT.mkdir(exist_ok=True)


def yahoo_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def _download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=" ".join(yahoo_symbol(s) for s in symbols),
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


def _extract(batch: pd.DataFrame, original: str, batch_size: int) -> pd.DataFrame:
    ys = yahoo_symbol(original)
    if batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns, pd.MultiIndex):
        level0 = set(map(str, batch.columns.get_level_values(0)))
        level1 = set(map(str, batch.columns.get_level_values(1)))
        if ys in level0:
            frame = batch[ys].copy()
        elif ys in level1:
            frame = batch.xs(ys, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if batch_size != 1:
            return pd.DataFrame()
        frame = batch.copy()
    frame.columns = [str(c).title() for c in frame.columns]
    frame.index = pd.to_datetime(frame.index, utc=True).normalize()
    return frame.sort_index()


def relerr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-9)


def main() -> None:
    with open(DATA / "prices.pkl", "rb") as f:
        prices = pickle.load(f)
    symbols = sorted(prices)
    all_dates = [bar[0] for bars in prices.values() for bar in bars]
    start = (pd.Timestamp(min(all_dates)).tz_convert("UTC") - pd.Timedelta(days=7)).date().isoformat()
    end = (pd.Timestamp(max(all_dates)).tz_convert("UTC") + pd.Timedelta(days=7)).date().isoformat()

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    chunk_size = 35
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        data = pd.DataFrame()
        for attempt in range(3):
            try:
                data = _download(chunk, start, end)
                if not data.empty:
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)
        for sym in chunk:
            frame = _extract(data, sym, len(chunk))
            if frame.empty or "Open" not in frame:
                failed.append(sym)
            else:
                frames[sym] = frame

    still_failed: list[str] = []
    for sym in failed:
        frame = pd.DataFrame()
        for attempt in range(3):
            try:
                data = _download([sym], start, end)
                frame = _extract(data, sym, 1)
                if not frame.empty and "Open" in frame:
                    break
            except Exception:
                pass
            time.sleep(2 ** attempt)
        if frame.empty or "Open" not in frame:
            still_failed.append(sym)
        else:
            frames[sym] = frame

    opens: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    rows: list[dict] = []
    total_bars = total_open = 0
    for sym, bars in prices.items():
        frame = frames.get(sym, pd.DataFrame())
        existing = pd.DataFrame(
            [(pd.Timestamp(t).tz_convert("UTC").normalize(), float(h), float(l), float(c))
             for t, h, l, c in bars],
            columns=["Date", "H_existing", "L_existing", "C_existing"],
        ).set_index("Date")
        total_bars += len(existing)
        if frame.empty:
            opens[sym] = []
            rows.append({"symbol": sym, "bars": len(existing), "opens": 0, "coverage": 0.0,
                         "scale": "missing", "median_close_relerr": np.nan,
                         "p99_hlc_relerr": np.nan})
            continue
        overlap = existing.join(frame, how="left")
        raw_close = overlap.get("Close")
        adj_close = overlap.get("Adj Close")
        valid_raw = raw_close.notna()
        raw_med = float(np.nanmedian(relerr(raw_close[valid_raw].to_numpy(),
                                             overlap.loc[valid_raw, "C_existing"].to_numpy()))) if valid_raw.any() else np.inf
        adj_med = np.inf
        if adj_close is not None:
            valid_adj = adj_close.notna()
            if valid_adj.any():
                adj_med = float(np.nanmedian(relerr(adj_close[valid_adj].to_numpy(),
                                                     overlap.loc[valid_adj, "C_existing"].to_numpy())))
        use_adjusted = adj_med + 1e-8 < raw_med
        selected = frame.copy()
        scale = "adjusted" if use_adjusted else "raw"
        if use_adjusted:
            factor = selected["Adj Close"] / selected["Close"]
            for col in ["Open", "High", "Low", "Close"]:
                selected[col] = selected[col] * factor
        selected = selected.replace([np.inf, -np.inf], np.nan)
        open_series = selected["Open"].dropna()
        opens[sym] = [(pd.Timestamp(idx).tz_convert("UTC"), float(val)) for idx, val in open_series.items()]
        matched = existing.join(selected[["Open", "High", "Low", "Close"]], how="left")
        have = matched["Open"].notna()
        total_open += int(have.sum())
        errs = []
        for ext, got in [("H_existing", "High"), ("L_existing", "Low"), ("C_existing", "Close")]:
            mask = matched[got].notna()
            if mask.any():
                errs.extend(relerr(matched.loc[mask, got].to_numpy(), matched.loc[mask, ext].to_numpy()).tolist())
        rows.append({
            "symbol": sym, "bars": len(existing), "opens": int(have.sum()),
            "coverage": float(have.mean()) if len(have) else 0.0,
            "scale": scale,
            "median_close_relerr": min(raw_med, adj_med),
            "p99_hlc_relerr": float(np.nanpercentile(errs, 99)) if errs else np.nan,
        })

    with open(DATA / "opens.pkl", "wb") as f:
        pickle.dump(opens, f, protocol=pickle.HIGHEST_PROTOCOL)
    audit = pd.DataFrame(rows).sort_values(["coverage", "symbol"])
    audit.to_csv(OUT / "open_alignment_by_symbol.csv", index=False)
    summary = {
        "symbols": len(symbols),
        "symbols_failed": still_failed,
        "total_existing_bars": total_bars,
        "total_aligned_opens": total_open,
        "bar_coverage": total_open / max(total_bars, 1),
        "symbols_below_95pct_coverage": audit.loc[audit.coverage < 0.95, "symbol"].tolist(),
        "median_symbol_close_relerr": float(audit["median_close_relerr"].replace(np.inf, np.nan).median()),
        "p99_symbol_hlc_relerr_median": float(audit["p99_hlc_relerr"].median()),
        "source": "Yahoo Finance via yfinance; raw or split/dividend-adjusted scale selected per symbol by HLC alignment",
    }
    (OUT / "open_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["bar_coverage"] < 0.97:
        raise SystemExit(f"Open coverage too low for a defensible audit: {summary['bar_coverage']:.2%}")


if __name__ == "__main__":
    main()
