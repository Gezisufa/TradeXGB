"""
Feature engineering for the BTC direction classifier.

All features at time t use only information available up to and including t.
The target is computed look-ahead by one bar, so the last row is always dropped.
"""
from __future__ import annotations

import logging
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ── indicator window constants ──────────────────────────────────────────────
EMA_FAST = 10
EMA_SLOW = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compute technical indicators and the next-day direction target.

    All features use only data available at time t (no look-ahead).
    Target: 1 if close[t+1] > close[t], else 0 — set via shift(-1).
    """
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    out[f"ema_{EMA_FAST}"] = ta.ema(out["Close"], length=EMA_FAST)
    out[f"ema_{EMA_SLOW}"] = ta.ema(out["Close"], length=EMA_SLOW)

    macd = ta.macd(out["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    out["macd_line"] = macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    out["macd_signal"] = macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    out["macd_hist"] = macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]

    out[f"rsi_{RSI_PERIOD}"] = ta.rsi(out["Close"], length=RSI_PERIOD)

    stoch = ta.stoch(out["High"], out["Low"], out["Close"], k=STOCH_K, d=STOCH_D)
    out["stoch_k"] = stoch[f"STOCHk_{STOCH_K}_{STOCH_D}_3"]
    out["stoch_d"] = stoch[f"STOCHd_{STOCH_K}_{STOCH_D}_3"]

    out[f"atr_{ATR_PERIOD}"] = ta.atr(out["High"], out["Low"], out["Close"], length=ATR_PERIOD)

    bb = ta.bbands(out["Close"], length=BB_PERIOD, std=BB_STD)
    upper_col = next(c for c in bb.columns if c.startswith("BBU"))
    lower_col = next(c for c in bb.columns if c.startswith("BBL"))
    mid_col = next(c for c in bb.columns if c.startswith("BBM"))
    out["bb_width"] = (bb[upper_col] - bb[lower_col]) / bb[mid_col]

    out["obv"] = ta.obv(out["Close"], out["Volume"])
    out["volume_pct_change"] = out["Volume"].pct_change()

    for lag in (1, 3, 5):
        out[f"ret_{lag}d"] = out["Close"].pct_change(lag)

    # shift(-1): label at t is the direction of t+1 — the last row has no label
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    # Drop the last row (no label) and any NaN rows from indicator warm-up
    out.drop(out.index[-1], inplace=True)

    feature_cols = [c for c in out.columns if c not in ("Open", "High", "Low", "Close", "Volume", "target")]
    out.dropna(subset=feature_cols + ["target"], inplace=True)

    X = out[feature_cols]
    y = out["target"]

    logger.info("Feature matrix: %d rows × %d features  (target balance: %.1f%% up)",
                len(X), X.shape[1], y.mean() * 100)
    return X, y


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and y chronologically (no shuffling) into train and test sets."""
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info("Train: %s → %s  (%d rows)", X_train.index[0].date(), X_train.index[-1].date(), len(X_train))
    logger.info("Test : %s → %s  (%d rows)", X_test.index[0].date(), X_test.index[-1].date(), len(X_test))
    return X_train, X_test, y_train, y_test
