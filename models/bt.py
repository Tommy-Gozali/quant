
import backtrader as bt

FAST        = 20
SLOW        = 50
class SMACross(bt.Strategy):
    """
    Long-only SMA crossover.
    BUY  all-in when fast SMA crosses above slow SMA.
    SELL all-out when fast SMA crosses below slow SMA.
    Uses next-bar market orders (no lookahead).
    """
    params = dict(fast=FAST, slow=SLOW)

    def __init__(self):
        self.fast_sma = bt.ind.SMA(self.data.close, period=self.p.fast)
        self.slow_sma = bt.ind.SMA(self.data.close, period=self.p.slow)
        self.crossup   = bt.ind.CrossUp(self.fast_sma, self.slow_sma)
        self.crossdown = bt.ind.CrossDown(self.fast_sma, self.slow_sma)
        self.order     = None
        self.trades    = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            direction = "BUY " if order.isbuy() else "SELL"
            self.trades.append({
                "bar":       len(self),
                "direction": direction,
                "price":     order.executed.price,
                "size":      order.executed.size,
                "value":     order.executed.value,
                "comm":      order.executed.comm,
            })
        self.order = None

    def next(self):
        if self.order:
            return   # wait for pending order

        if not self.position:
            if self.crossup[0]:
                size = int(self.broker.getcash() / self.data.close[0])
                self.order = self.buy(size=size)
        else:
            if self.crossdown[0]:
                self.order = self.sell(size=self.position.size)
