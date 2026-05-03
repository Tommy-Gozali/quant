"""
Full SPY Trading Workflow
==========================
Stage 0 - Fetch real OHLCV data from Yahoo Finance
Stage 1 - In-sample run: SMA(20/50) crossover + stats
Stage 2 - Permutation test: does the signal have real edge?
Stage 3 - Out-of-sample validation
Stage 4 - Combined performance chart
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
from config.config_loader import TICKER, N_IN, N_OUT, CASH, COMMISSION, RF_DAILY, N_PERMS, FAST, SLOW
from data_management.fetcher.yahoo_finance import YahooFinanceFetcher
from models.sma_cross import SMACrossSignal

warnings.filterwarnings("ignore")  # suppress yfinance/pandas noise
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════
# STAGE 0 - FETCH OHLCV DATA FROM YAHOO FINANCE
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"STAGE 0 - Fetching {TICKER} OHLCV data from Yahoo Finance")
print("=" * 60)



fetcher = YahooFinanceFetcher(TICKER)
df_in, lr_in, px_in, df_out, lr_out, px_out = fetcher.fetch()

strategy = SMACrossSignal()

print(f"In-sample    : {N_IN} bars | {df_in.index[0].date()} to {df_in.index[-1].date()}")
print(f"               start=${px_in[0]:.2f}  end=${px_in[-1]:.2f}")
print(f"Out-of-sample: {N_OUT} bars | {df_out.index[0].date()} to {df_out.index[-1].date()}")
print(f"               start=${px_out[0]:.2f}  end=${px_out[-1]:.2f}")

# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def sharpe(lr):
    ann_r = lr.mean() * 252
    ann_v = lr.std()  * np.sqrt(252)
    return (ann_r - RF_DAILY * 252) / ann_v if ann_v > 0 else 0

def dd_curve(vals):
    pk = np.maximum.accumulate(vals)
    return (vals - pk) / pk * 100

def _max_dd_duration(port):
    peak, cur, worst = port[0], 0, 0
    for v in port:
        if v >= peak:
            peak, cur = v, 0
        else:
            cur += 1
            worst = max(worst, cur)
    return worst


def run_strategy(df, cash=CASH, commission=COMMISSION, printlog=False):
    """
    Simulate the SMA crossover strategy on a given OHLCV DataFrame.

    Uses integer share sizing and applies commission on both entry and exit,
    matching realistic brokerage behaviour.

    Returns
    -------
    port : np.ndarray
        Portfolio value at the close of every bar (length == len(df)).
    final_val : float
        Portfolio value on the last bar.
    total_ret : float
        Total return as a percentage.
    trades : list[dict]
        One entry per order execution with keys:
        bar, direction, price, size, value, comm.
    """
    prices = df["close"].values
    pos    = strategy.signal(pd.Series(prices))

    port   = [cash]
    cash_  = cash
    units  = 0
    in_pos = False
    trades = []

    for i in range(1, len(prices)):
        p = prices[i]
        if pos.iloc[i] == 1 and not in_pos:
            units = int(cash_ / p)
            if units > 0:
                cost   = units * p * commission
                cash_ -= units * p + cost
                in_pos = True
                trades.append({"bar": i, "direction": "BUY ", "price": p,
                                "size": units, "value": units * p, "comm": cost})
        elif pos.iloc[i] == 0 and in_pos:
            proceeds = units * p
            cost     = proceeds * commission
            cash_   += proceeds - cost
            trades.append({"bar": i, "direction": "SELL", "price": p,
                            "size": units, "value": proceeds, "comm": cost})
            units  = 0
            in_pos = False
        port.append(cash_ + units * p)

    port      = np.array(port)
    final_val = port[-1]
    total_ret = (final_val - cash) / cash * 100

    if printlog:
        print(f"\n  Starting cash    : ${cash:,.0f}")
        print(f"  Final value      : ${final_val:,.0f}")
        print(f"  Total return     : {total_ret:.2f}%")

        port_lr = np.log(port[1:] / np.maximum(port[:-1], 1e-10))
        print(f"  Sharpe ratio     : {sharpe(port_lr):.3f}")
        print(f"  Max drawdown     : {dd_curve(port).min():.2f}%")
        print(f"  Max DD duration  : {_max_dd_duration(port)} bars")

        buys   = [t for t in trades if t["direction"] == "BUY "]
        sells  = [t for t in trades if t["direction"] == "SELL"]
        pairs  = list(zip(buys, sells))
        pnls   = [s["value"] - s["comm"] - b["value"] - b["comm"] for b, s in pairs]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        n      = len(pairs)
        wr     = len(wins) / n * 100 if n else 0
        print(f"  Total trades     : {n}")
        print(f"  Win rate         : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
        if wins and losses:
            print(f"  Avg win / loss   : ${np.mean(wins):,.0f} / ${np.mean(losses):,.0f}")

    return port, final_val, total_ret, trades


# ══════════════════════════════════════════════════════════════════
# STAGE 1 - IN-SAMPLE RUN
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 1 - In-sample run")
print("=" * 60)

port_in, final_in, ret_in, trades_in = run_strategy(df_in, printlog=True)

# Buy-and-hold benchmark (in-sample)
bh_in_ret = (px_in[-1] / px_in[0] - 1) * 100
print(f"\n  Buy-and-hold     : {bh_in_ret:.2f}%  (no commission)")
print(f"  Strategy edge    : {ret_in - bh_in_ret:+.2f}%")

# ══════════════════════════════════════════════════════════════════
# STAGE 2 - PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STAGE 2 - Permutation test ({N_PERMS:,} shuffles)")
print("=" * 60)

actual_lr     = strategy.apply(lr_in)
actual_sharpe = sharpe(actual_lr)

rng = np.random.default_rng(0)
perm_sharpes = np.array([
    sharpe(strategy.apply(rng.permutation(lr_in)))
    for _ in range(N_PERMS)
])

p_value  = (perm_sharpes >= actual_sharpe).mean()
pct_rank = (perm_sharpes < actual_sharpe).mean() * 100
pct_95   = np.percentile(perm_sharpes, 95)

print(f"\n  Actual strategy Sharpe   : {actual_sharpe:.3f}")
print(f"  Permutation mean Sharpe  : {perm_sharpes.mean():.3f}")
print(f"  Permutation 95th pctile  : {pct_95:.3f}")
print(f"  p-value                  : {p_value:.3f}")
print(f"  Percentile rank          : {pct_rank:.1f}th")
verdict = "EDGE DETECTED" if p_value < 0.05 else "NO SIGNIFICANT EDGE"
print(f"\n  RESULT: {verdict}  (p={p_value:.3f})")

# ══════════════════════════════════════════════════════════════════
# STAGE 3 - OUT-OF-SAMPLE VALIDATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 3 - Out-of-sample validation")
print("=" * 60)

port_out, final_out, ret_out, trades_out = run_strategy(df_out, printlog=True)

bh_out_ret = (px_out[-1] / px_out[0] - 1) * 100
print(f"\n  Buy-and-hold     : {bh_out_ret:.2f}%  (no commission)")
print(f"  Strategy edge    : {ret_out - bh_out_ret:+.2f}%")

# Trade log
print("\n  Trade log:")
print(f"  {'#':<4} {'Dir':<5} {'Bar':<6} {'Price':>8} {'Size':>7} {'Comm':>8}")
print("  " + "-"*45)
for i, t in enumerate(trades_out, 1):
    print(f"  {i:<4} {t['direction']:<5} {t['bar']:<6} "
          f"${t['price']:>7.2f} {int(t['size']):>7} ${t['comm']:>7.2f}")

# ══════════════════════════════════════════════════════════════════
# STAGE 4 - COMBINED PERFORMANCE CHART
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STAGE 4 - Generating charts")
print("=" * 60)

bh_in_curve  = CASH * px_in  / px_in[0]
bh_out_curve = CASH * px_out / px_out[0]

fig = plt.figure(figsize=(16, 18))
fig.suptitle(
    f"SPY SMA({FAST}/{SLOW}) Workflow\n"
    "Stage 1: In-sample  |  Stage 2: Permutation Test  |  Stage 3: Out-of-sample",
    fontsize=13, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

days_in  = np.arange(N_IN)
days_out = np.arange(N_OUT)

# Panel 1: In-sample price + SMAs
ax1 = fig.add_subplot(gs[0, :])
fma_in = pd.Series(px_in).rolling(FAST).mean()
sma_in = pd.Series(px_in).rolling(SLOW).mean()
sig_in = strategy.signal(pd.Series(px_in))

ax1.plot(days_in, px_in, color="black", linewidth=0.9, alpha=0.7, label="SPY")
ax1.plot(days_in, fma_in.values, color="royalblue",  linewidth=1.3, label=f"SMA({FAST})")
ax1.plot(days_in, sma_in.values, color="darkorange",  linewidth=1.3, label=f"SMA({SLOW})")
prev = 0
for i in range(1, N_IN):
    if sig_in.iloc[i] == 1 and prev == 0:
        start = i
        prev = 1
    elif sig_in.iloc[i] == 0 and prev == 1:
        ax1.axvspan(start, i, color="green", alpha=0.07); prev = 0
if prev == 1:
    ax1.axvspan(start, N_IN, color="green", alpha=0.07)

ax1.set_title("Stage 1 - In-sample: Price + SMA crossover (green = long position)")
ax1.set_xlabel("Trading Days")
ax1.set_ylabel("Price ($)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# Panel 2: Permutation test histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(perm_sharpes, bins=50, color="steelblue", alpha=0.7,
         edgecolor="white", label=f"Permuted ({N_PERMS:,} shuffles)")
ax2.axvline(actual_sharpe, color="red", linewidth=2.0,
            label=f"Actual: {actual_sharpe:.3f}")
ax2.axvline(pct_95, color="orange", linewidth=1.5, linestyle="--",
            label=f"95th pctile: {pct_95:.3f}")
ax2.set_title(f"Stage 2 - Permutation Test  (p={p_value:.3f})")
ax2.set_xlabel("Sharpe Ratio")
ax2.set_ylabel("Count")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

# Panel 3: In-sample portfolio value
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(port_in,     color="royalblue", linewidth=1.8, label="Strategy (w/ commission)")
ax3.plot(bh_in_curve, color="gray",      linewidth=1.2, linestyle="--",
         label="Buy-and-hold", alpha=0.8)
ax3.axhline(CASH, color="black", linestyle=":", linewidth=0.8)
ax3.set_title(f"Stage 1 - In-sample Performance  (strat: {ret_in:.1f}%  BH: {bh_in_ret:.1f}%)")
ax3.set_xlabel("Trading Days")
ax3.set_ylabel("Portfolio Value ($)")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.2)

# Panel 4: Out-of-sample price + entries/exits
ax4 = fig.add_subplot(gs[2, :])
fma_out = pd.Series(px_out).rolling(FAST).mean()
sma_out = pd.Series(px_out).rolling(SLOW).mean()
sig_out = strategy.signal(pd.Series(px_out))

ax4.plot(days_out, px_out, color="black", linewidth=0.9, alpha=0.7, label="SPY")
ax4.plot(days_out, fma_out.values, color="royalblue",  linewidth=1.2, label=f"SMA({FAST})")
ax4.plot(days_out, sma_out.values, color="darkorange",  linewidth=1.2, label=f"SMA({SLOW})")

sig_arr = sig_out.values
entries = np.where((sig_arr[1:]==1) & (sig_arr[:-1]==0))[0] + 1
exits   = np.where((sig_arr[1:]==0) & (sig_arr[:-1]==1))[0] + 1

if len(entries):
    ax4.scatter(entries, px_out[entries], marker="^", color="green", s=120,
                zorder=5, label=f"Entry (n={len(entries)})")
if len(exits):
    ax4.scatter(exits, px_out[exits], marker="v", color="red", s=120,
                zorder=5, label=f"Exit  (n={len(exits)})")

prev = 0
for i in range(1, N_OUT):
    if sig_out.iloc[i] == 1 and prev == 0:
        start = i
        prev = 1
    elif sig_out.iloc[i] == 0 and prev == 1:
        ax4.axvspan(start, i, color="green", alpha=0.07)
        prev = 0
if prev == 1:
    ax4.axvspan(start, N_OUT, color="green", alpha=0.07)

ax4.set_title("Stage 3 - Out-of-sample: Entry/Exit signals + invested periods (green)")
ax4.set_xlabel("Trading Days")
ax4.set_ylabel("Price ($)")
ax4.legend(fontsize=9)
ax4.grid(alpha=0.2)

# Panel 5: Out-of-sample portfolio value
ax5 = fig.add_subplot(gs[3, 0])
ax5.plot(port_out,     color="royalblue", linewidth=1.8, label="Strategy (w/ commission)")
ax5.plot(bh_out_curve, color="gray",      linewidth=1.2, linestyle="--",
         label="Buy-and-hold", alpha=0.8)
ax5.axhline(CASH, color="black", linestyle=":", linewidth=0.8)
ax5.set_title(f"Stage 3 - Out-of-sample Performance  (strat: {ret_out:.1f}%  BH: {bh_out_ret:.1f}%)")
ax5.set_xlabel("Trading Days")
ax5.set_ylabel("Portfolio Value ($)")
ax5.legend(fontsize=9)
ax5.grid(alpha=0.2)

# Panel 6: Out-of-sample drawdown
ax6 = fig.add_subplot(gs[3, 1])
ax6.fill_between(days_out, dd_curve(port_out),     color="royalblue", alpha=0.45,
                 label="Strategy")
ax6.fill_between(days_out, dd_curve(bh_out_curve), color="gray",      alpha=0.35,
                 label="Buy-and-hold")
ax6.set_title("Stage 3 - Out-of-sample Drawdown (%)")
ax6.set_xlabel("Trading Days")
ax6.set_ylabel("Drawdown (%)")
ax6.legend(fontsize=9)
ax6.grid(alpha=0.2)

plt.savefig("bt_workflow.png", dpi=150, bbox_inches="tight")
print("\nChart saved.")
print("\nDone.")