import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Regenerate Dummy Data (Pandas)
num_days = 500
sectors = ["Tech", "Finance", "Energy", "Health", "Retail"]
dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(num_days)]

data_signals = []
for d in dates:
    for s in sectors:
        data_signals.append({"sector": s, "date": d, "value": np.random.uniform(-1, 1)})
df_signals = pd.DataFrame(data_signals)

data_returns = []
for d in dates:
    for s in sectors:
        data_returns.append({"date": d, "sector": s, "total_return": np.random.normal(0.0002, 0.01)})
df_returns = pd.DataFrame(data_returns)

# 2. Join and Strategy Calculation
df_combined = pd.merge(df_signals, df_returns, on=["date", "sector"])
df_combined["weighted_return"] = df_combined["value"] * df_combined["total_return"]

# Daily strategy return
df_daily_pnl = df_combined.groupby("date")["weighted_return"].sum().reset_index()
df_daily_pnl = df_daily_pnl.sort_values("date")

# 3. Rolling Sharpe Ratio Calculation (e.g., 60-day window)
window = 60
rolling_mean = df_daily_pnl["weighted_return"].rolling(window=window).mean()
rolling_std = df_daily_pnl["weighted_return"].rolling(window=window).std()
df_daily_pnl["rolling_sharpe"] = (rolling_mean / rolling_std) * np.sqrt(252)

# 4. Sector-wise Sharpe Ratio
sector_returns = df_combined.groupby("sector")["weighted_return"].agg(["mean", "std"])
sector_returns["sharpe"] = (sector_returns["mean"] / sector_returns["std"]) * np.sqrt(252)
sector_returns = sector_returns.sort_values("sharpe", ascending=False)

# 5. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Rolling Sharpe Plot
ax1.plot(df_daily_pnl["date"], df_daily_pnl["rolling_sharpe"], color='teal', label=f"{window}-Day Rolling Sharpe")
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_title(f"Rolling Strategy Sharpe Ratio ({window}-Day Window)")
ax1.set_ylabel("Annualized Sharpe Ratio")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Sector-wise Sharpe Plot (Bar chart)
sector_returns["sharpe"].plot(kind='bar', ax=ax2, color='skyblue', edgecolor='navy')
ax2.set_title("Annualized Sharpe Ratio by Sector")
ax2.set_ylabel("Sharpe Ratio")
ax2.set_xlabel("Sector")
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("sharpe_ratio_plots.png")

# Prepare summary for response
print(sector_returns["sharpe"])
