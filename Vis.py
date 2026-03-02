import polars as pl
import matplotlib.pyplot as plt

# 1. Join and Calculate Weighted Returns per Sector
df_combined = df_signals.join(df_returns, on=["date", "sector"])
df_combined = df_combined.with_columns(
    (pl.col("value") * pl.col("total_return")).alias("weighted_return")
)

# 2. Calculate Cumulative PnL for Each Individual Sector
df_combined = df_combined.sort(["sector", "date"]).with_columns(
    (1 + pl.col("weighted_return")).cum_prod().over("sector").alias("cum_pnl_sector")
)

# 3. Aggregate for Overall Strategy PnL
df_strategy = (
    df_combined.group_by("date")
    .agg(pl.col("weighted_return").sum().alias("daily_strategy_return"))
    .sort("date")
    .with_columns(
        (1 + pl.col("daily_strategy_return")).cum_prod().alias("cum_pnl_total")
    )
)

# 4. Plotting
plt.figure(figsize=(12, 6))

# Plot each sector
for sector in df_combined["sector"].unique():
    sector_data = df_combined.filter(pl.col("sector") == sector)
    plt.plot(sector_data["date"], sector_data["cum_pnl_sector"], alpha=0.4, label=f"{sector}")

# Plot overall strategy in bold
plt.plot(df_strategy["date"], df_strategy["cum_pnl_total"], color="black", linewidth=2.5, label="Overall Strategy")

plt.title("Cumulative PnL: Overall Strategy vs. Individual Sectors")
plt.ylabel("Growth of $1")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
