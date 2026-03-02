import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Generate Dummy Data
num_days = 500
sectors = ["Tech", "Finance", "Energy", "Health", "Retail"]
dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(num_days)]

# Create signals: Sector, Date, Value
# Signals range from -1 to 1
data_signals = []
for d in dates:
    for s in sectors:
        data_signals.append({"sector": s, "date": d, "value": np.random.uniform(-1, 1)})
df_signals = pl.DataFrame(data_signals)

# Create returns: Date, Sector, total_return
# Returns roughly centered around 0 with some volatility
data_returns = []
for d in dates:
    for s in sectors:
        data_returns.append({"date": d, "sector": s, "total_return": np.random.normal(0.0005, 0.01)})
df_returns = pl.DataFrame(data_returns)

# 2. Join and Calculate PnL
# We assume signal at day t captures return at day t (or we might need to shift)
# Standard backtest: Signal at end of day t-1 is used for return of day t.
# For simplicity, let's assume 'value' is the weight we want to hold for the period starting at 'date'
# and 'total_return' is the return for that same period.

df_combined = df_signals.join(df_returns, on=["date", "sector"])

# Calculate weighted return per sector
df_combined = df_combined.with_columns(
    (pl.col("value") * pl.col("total_return")).alias("weighted_return")
)

# Aggregate daily strategy returns
df_daily_pnl = df_combined.group_by("date").agg(
    pl.col("weighted_return").sum().alias("strategy_return")
).sort("date")

# 3. Calculate Metrics
# Cumulative PnL
df_daily_pnl = df_daily_pnl.with_columns(
    (1 + pl.col("strategy_return")).cum_prod().alias("cumulative_return")
)

# Sharpe Ratio
# Assume 252 trading days
avg_return = df_daily_pnl["strategy_return"].mean()
std_return = df_daily_pnl["strategy_return"].std()
sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return != 0 else 0

# Max Drawdown
cum_ret = df_daily_pnl["cumulative_return"].to_numpy()
running_max = np.maximum.accumulate(cum_ret)
drawdown = (cum_ret - running_max) / running_max
max_drawdown = drawdown.min()

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot Cumulative Return
ax1.plot(df_daily_pnl["date"], df_daily_pnl["cumulative_return"], label="Strategy Cumulative Return", color='blue')
ax1.set_title("Strategy Performance: Cumulative Return")
ax1.set_ylabel("Growth of $1")
ax1.grid(True)
ax1.legend()

# Plot Drawdown
ax2.fill_between(df_daily_pnl["date"], drawdown, 0, color='red', alpha=0.3, label="Drawdown")
ax2.set_title("Strategy Drawdown")
ax2.set_ylabel("Drawdown %")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("strategy_performance.png")

print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Annualized Volatility: {std_return * np.sqrt(252):.2%}")
print(f"Total Return: {(df_daily_pnl['cumulative_return'][-1] - 1):.2%}")
