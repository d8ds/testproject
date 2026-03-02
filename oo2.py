import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Ensure sorted
signal_df = signal_df.sort(["sector", "date"])
df_returns = df_returns.sort(["sector", "date"])

# Shift returns backward (so signal_t uses return_t+1)
df_returns = df_returns.with_columns(
    pl.col("total_return")
      .shift(-1)
      .over("sector")
      .alias("fwd_return")
)

# Join
df = signal_df.join(
    df_returns.select(["date", "sector", "fwd_return"]),
    on=["date", "sector"],
    how="inner"
).drop_nulls()

df = df.with_columns(
    (
        pl.col("value") -
        pl.col("value").mean().over("date")
    ).alias("signal_demeaned")
)

# scale to 1 gross leverage
df = df.with_columns(
    (
        pl.col("signal_demeaned") /
        pl.col("signal_demeaned").abs().sum().over("date")
    ).alias("weight")
)

df = df.with_columns(
    (pl.col("weight") * pl.col("fwd_return")).alias("pnl")
)

daily_pnl = df.group_by("date").agg(
    pl.col("pnl").sum().alias("daily_pnl")
).sort("date")

returns = daily_pnl["daily_pnl"].to_numpy()

mean_ret = np.mean(returns)
std_ret = np.std(returns)

# Annualization (assume 252 trading days)
sharpe = np.sqrt(252) * mean_ret / std_ret

cumulative = np.cumprod(1 + returns)

max_dd = np.max(np.maximum.accumulate(cumulative) - cumulative)
max_dd_pct = max_dd / np.maximum.accumulate(cumulative).max()

print("Sharpe:", sharpe)
print("Mean daily return:", mean_ret)
print("Vol:", std_ret)
print("Max Drawdown:", max_dd_pct)

plt.figure()
plt.plot(daily_pnl["date"], cumulative)
plt.title("Cumulative PnL")
plt.xticks(rotation=45)
plt.show()

df = df.with_columns(
    pl.col("weight").shift(1).over("sector").alias("prev_weight")
)

turnover = df.with_columns(
    (pl.col("weight") - pl.col("prev_weight")).abs()
).group_by("date").agg(
    pl.col("weight").sum().alias("daily_turnover")
)

avg_turnover = turnover["daily_turnover"].mean()
print("Average turnover:", avg_turnover)
