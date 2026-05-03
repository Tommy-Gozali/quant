import time
import yfinance as yf
import numpy as np
import pandas as pd
from config.config_loader import N_IN, N_OUT
class YahooFinanceFetcher:
    """
    Download daily OHLCV data for a single ticker from Yahoo Finance and split
    it into chronological in-sample and out-of-sample windows.

    The split is always taken from the *most recent* available history so that
    out-of-sample data is as current as possible.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "SPY", "AAPL", "BTC-USD").
    n_in : int
        Number of trading bars in the in-sample window. Default: ``N_IN`` from
        config (~5 years at 252 bars/year).
    n_out : int
        Number of trading bars in the out-of-sample window. Default: ``N_OUT``
        from config (~2 years).
    retries : int
        Maximum download attempts before raising ``RuntimeError``. Default: 3.
    retry_delay : float
        Seconds to wait between failed download attempts. Default: 2.0.
    """

    def __init__(
        self,
        ticker: str,
        n_in: int = N_IN,
        n_out: int = N_OUT,
        retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.ticker      = ticker
        self.n_in        = n_in
        self.n_out       = n_out
        self.retries     = retries
        self.retry_delay = retry_delay

    def fetch(self) -> tuple:
        """
        Download and split OHLCV data into in-sample and out-of-sample sets.

        Internally fetches ``n_in + n_out + 1`` bars (the extra bar is an
        anchor used to compute the first log-return without look-ahead), then
        drops it so every returned array starts at the same calendar date.

        Returns
        -------
        df_in : pd.DataFrame, shape (n_in, 6)
            In-sample OHLCV bars with columns
            ``[open, high, low, close, volume, openinterest]`` and a
            ``DatetimeIndex``. Ready to pass directly to
            ``bt.feeds.PandasData``.
        lr_in : np.ndarray, shape (n_in,)
            Daily log-returns for the in-sample period.
            ``lr_in[i] = log(close[i] / close[i-1])``.
        px_in : np.ndarray, shape (n_in,)
            Adjusted close prices for the in-sample period.
        df_out : pd.DataFrame, shape (n_out, 6)
            Out-of-sample OHLCV bars, same schema as ``df_in``.
        lr_out : np.ndarray, shape (n_out,)
            Daily log-returns for the out-of-sample period.
        px_out : np.ndarray, shape (n_out,)
            Adjusted close prices for the out-of-sample period.

        Raises
        ------
        ValueError
            If Yahoo Finance returns fewer bars than ``n_in + n_out + 1``.
        RuntimeError
            If all download attempts fail.
        """
        needed       = self.n_in + self.n_out + 1  # +1 anchor bar for first log-return
        years_needed = max(int(np.ceil(needed / 252)) + 1, 2)

        raw = self._download(years_needed)
        raw = self._normalise(raw)

        if len(raw) < needed:
            raise ValueError(
                f"Only {len(raw)} bars available for '{self.ticker}'; "
                f"need at least {needed} (n_in={self.n_in} + n_out={self.n_out} + 1 anchor)"
            )

        raw       = raw.iloc[-needed:]
        close_all = raw["close"].values                       # length: needed
        lr_all    = np.log(close_all[1:] / close_all[:-1])   # length: n_in + n_out
        bars_all  = raw.iloc[1:].copy()                       # length: n_in + n_out

        df_in  = bars_all.iloc[:self.n_in]
        df_out = bars_all.iloc[self.n_in:]
        px_in  = df_in["close"].values
        px_out = df_out["close"].values
        lr_in  = lr_all[:self.n_in]
        lr_out = lr_all[self.n_in:]

        return df_in, lr_in, px_in, df_out, lr_out, px_out

    def _download(self, years: int) -> pd.DataFrame:
        """
        Attempt to download ``years`` years of daily data from Yahoo Finance,
        retrying up to ``self.retries`` times on failure.

        Parameters
        ----------
        years : int
            Number of calendar years of history to request.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame as returned by ``yf.download``.

        Raises
        ------
        RuntimeError
            If every attempt raises an exception or returns empty data.
        """
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
    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise column names and schema from a raw ``yf.download`` result.

        Handles both the flat column format of older yfinance versions and the
        ``(Price, Ticker)`` MultiIndex introduced in newer releases. Retains
        only ``[open, high, low, close, volume]``, drops rows with any NaN,
        and appends an ``openinterest`` column (always 0) required by
        ``bt.feeds.PandasData``.

        Parameters
        ----------
        raw : pd.DataFrame
            DataFrame as returned by ``yf.download``.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with lowercase columns and ``openinterest``.
        """
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        raw["openinterest"] = 0.0
        return raw

