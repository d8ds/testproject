# TODO
import polars as pl
import numpy as np

# 1. Join signal and returns
# We assume 'value' in df_signals is the weight/exposure for that sector on that date
df_combined = df_signals.join(df_returns, on=["date", "sector"])

# 2. Calculate daily strategy return (Signal * Return)
df_daily_pnl = (
    df_combined
    .with_columns(
        (pl.col("value") * pl.col("total_return")).alias("weighted_return")
    )
    .group_by("date")
    .agg(pl.col("weighted_return").sum().alias("strategy_return"))
    .sort("date")
)

# 3. Calculate Cumulative Returns
df_daily_pnl = df_daily_pnl.with_columns(
    (1 + pl.col("strategy_return")).cum_prod().alias("cumulative_return")
)

# 4. Calculate Sharpe Ratio (Annualized)
stats = df_daily_pnl.select([
    pl.col("strategy_return").mean().alias("avg_ret"),
    pl.col("strategy_return").std().alias("std_ret")
])
sharpe = (stats["avg_ret"][0] / stats["std_ret"][0]) * np.sqrt(252)

# 5. Max Drawdown
df_daily_pnl = df_daily_pnl.with_columns(
    pl.col("cumulative_return").cum_max().alias("rolling_max")
)
df_daily_pnl = df_daily_pnl.with_columns(
    ((pl.col("cumulative_return") - pl.col("rolling_max")) / pl.col("rolling_max")).alias("drawdown")
)
max_drawdown = df_daily_pnl["drawdown"].min()

# Shift signals by 1 day so today's return is multiplied by yesterday's signal
df_signals = df_signals.with_columns(pl.col("value").shift(1).over("sector"))
