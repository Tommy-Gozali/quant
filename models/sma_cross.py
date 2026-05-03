import numpy as np
import pandas as pd
from config.config_loader import FAST, SLOW


class SMACrossSignal:
    """
    Pure numpy/pandas implementation of a long-only SMA crossover signal.

    This is the single source of truth for the crossover logic used by:
    - the permutation test (via ``apply``)
    - the portfolio simulation (via ``signal``)
    - the Backtrader adapter (``models.bt.SMACross`` reads the same params)
    - chart panels (via ``signal``)

    Parameters
    ----------
    fast : int
        Lookback period for the fast SMA. Default: ``FAST`` from config.
    slow : int
        Lookback period for the slow SMA. Default: ``SLOW`` from config.
    """

    def __init__(self, fast: int = FAST, slow: int = SLOW):
        self.fast = fast
        self.slow = slow

    def signal(self, prices: pd.Series) -> pd.Series:
        """
        Compute a next-bar position series from a price series.

        Parameters
        ----------
        prices : pd.Series
            Sequence of close prices (any index).

        Returns
        -------
        pd.Series
            Float series aligned with ``prices``: ``1.0`` = long, ``0.0`` = flat.
            Shifted forward by 1 bar so today's position is determined by
            yesterday's SMA values (no lookahead).
        """
        fast_sma = prices.rolling(self.fast).mean()
        slow_sma = prices.rolling(self.slow).mean()
        return (fast_sma > slow_sma).astype(float).shift(1).fillna(0)

    def apply(self, log_returns: np.ndarray) -> np.ndarray:
        """
        Apply the signal to a log-return series.

        Reconstructs a relative price series via ``cumsum``, derives the
        position with ``signal``, then masks returns to zero on flat days.
        Used by the permutation test where only relative price movements
        matter, not the absolute price level.

        Parameters
        ----------
        log_returns : np.ndarray, shape (n,)
            Daily log-returns.

        Returns
        -------
        np.ndarray, shape (n,)
            Strategy log-returns: equal to ``log_returns`` on long days,
            ``0.0`` on flat days.
        """
        prices = pd.Series(np.exp(np.cumsum(log_returns)))
        position = self.signal(prices)
        return log_returns * position.values
