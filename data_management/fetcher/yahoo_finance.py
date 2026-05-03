import time
import yfinance as yf
import numpy as np
import pandas as pd
from config.config_loader import N_IN, N_OUT
class YahooFinanceFetcher:
    """Fetches and splits OHLCV data from Yahoo Finance into in/out-of-sample sets."""

    def __init__(self, ticker, n_in=N_IN, n_out=N_OUT, retries=3, retry_delay=2.0):
        self.ticker      = ticker
        self.n_in        = n_in
        self.n_out       = n_out
        self.retries     = retries
        self.retry_delay = retry_delay

    def fetch(self):
        """Returns (df_in, lr_in, px_in, df_out, lr_out, px_out)."""
        needed       = self.n_in + self.n_out + 1  # +1 anchor bar for first log-return
        years_needed = max(int(np.ceil(needed / 252)) + 1, 2)

        raw = self._download(years_needed)
        raw = self._normalise(raw)

        if len(raw) < needed:
            raise ValueError(
                f"Only {len(raw)} bars available for '{self.ticker}'; "
                f"need at least {needed} (n_in={self.n_in} + n_out={self.n_out} + 1 anchor)"
            )

        raw         = raw.iloc[-needed:]
        close_all   = raw["close"].values                           # length: needed
        lr_all      = np.log(close_all[1:] / close_all[:-1])       # length: n_in + n_out
        bars_all    = raw.iloc[1:].copy()                           # length: n_in + n_out

        df_in  = bars_all.iloc[:self.n_in]
        df_out = bars_all.iloc[self.n_in:]
        px_in  = df_in["close"].values
        px_out = df_out["close"].values
        lr_in  = lr_all[:self.n_in]
        lr_out = lr_all[self.n_in:]

        return df_in, lr_in, px_in, df_out, lr_out, px_out

    def _download(self, years):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                raw = yf.download(
                    self.ticker,
                    period=f"{years}y",
                    auto_adjust=True,
                    progress=False,
                    actions=False,
                )
                if raw is None or raw.empty:
                    raise ValueError(f"No data returned for ticker '{self.ticker}'")
                return raw
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    print(f"  [attempt {attempt}] {exc} — retrying in {self.retry_delay}s")
                    time.sleep(self.retry_delay)
        raise RuntimeError(
            f"Failed to fetch '{self.ticker}' after {self.retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _normalise(raw):
        # Flatten MultiIndex produced by newer yfinance versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        raw["openinterest"] = 0.0
        return raw

