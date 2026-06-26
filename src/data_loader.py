"""
Download and cache BTC-USD daily OHLCV data from Yahoo Finance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKER = "BTC-USD"
DEFAULT_PERIOD = "5y"
DEFAULT_SAVE_PATH = Path(__file__).parent.parent / "data" / "btc_daily.csv"


def download_btc(
    period: str = DEFAULT_PERIOD,
    save_path: Path | str = DEFAULT_SAVE_PATH,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Download BTC-USD daily OHLCV from Yahoo Finance and cache to CSV."""
    save_path = Path(save_path)

    if save_path.exists() and not force_reload:
        logger.info("Loading cached data from %s", save_path)
        df = pd.read_csv(save_path, index_col="Date", parse_dates=True)
        logger.info("Loaded %d rows (%s → %s)", len(df), df.index[0].date(), df.index[-1].date())
        return df

    logger.info("Downloading %s data (period=%s) …", TICKER, period)
    raw = yf.download(TICKER, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")

    # Keep only the OHLCV columns we need
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"

    # Drop any fully-null rows (public-holiday gaps)
    before = len(df)
    df.dropna(how="all", inplace=True)
    if len(df) < before:
        logger.warning("Dropped %d fully-null rows", before - len(df))

    # Forward-fill remaining sparse NaNs (e.g. missing volume on weekends)
    df.ffill(inplace=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    logger.info("Saved %d rows to %s", len(df), save_path)
    return df


def load_btc(save_path: Path | str = DEFAULT_SAVE_PATH) -> pd.DataFrame:
    return download_btc(save_path=save_path, force_reload=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = download_btc(force_reload=True)
    print(data.tail())
