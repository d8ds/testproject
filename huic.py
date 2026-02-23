import polars as pl
import numpy as np
from dataclasses import dataclass

# ─────────────────────────────────────────
# 假设你还有一个 etf_returns df:
# columns: factset_entity_id, date, ret (日收益率)
# ─────────────────────────────────────────

@dataclass
class BacktestConfig:
    signal_lag: int = 1          # 信号滞后几天入场，避免look-ahead
    holding_period: int = 5      # 持仓天数
    long_short: bool = True      # True=多空，False=纯多头
    n_quantiles: int = 5         # 分组数
    min_coverage: float = 0.1    # 最低holdings覆盖率
    transaction_cost: float = 0.001  # 单边手续费

# 1 preprocessing
def prepare_signal(etf_signal: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    标准化信号 + 滞后处理
    """
    signal = (
        etf_signal
        .filter(pl.col("coverage") > config.min_coverage)
        .sort(["factset_entity_id", "date"])
    )

    # 截面z-score标准化（每天在所有ETF间标准化）
    signal = signal.with_columns(
        pl.col("etf_sentiment")
          .map_batches(lambda x: (x - x.mean()) / (x.std() + 1e-8))
          .over("date")
          .alias("signal_zscore")
    )

    # 截面分位数（用于分组回测）
    signal = signal.with_columns(
        pl.col("signal_zscore")
          .map_batches(lambda x: pl.Series(
              pd.qcut(x.to_numpy(), config.n_quantiles, labels=False, duplicates="drop")
          ))
          .over("date")
          .alias("quantile")
    )

    # 信号滞后，确保t日信号用t+lag日的收益来计算
    signal = signal.with_columns(
        pl.col("signal_zscore").shift(config.signal_lag).over("factset_entity_id").alias("signal_lagged"),
        pl.col("quantile").shift(config.signal_lag).over("factset_entity_id").alias("quantile_lagged"),
    )

    return signal

# compute weights
def build_positions(signal: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    """
    基于信号构建每日仓位权重
    支持 holding_period > 1 的换仓逻辑（overlapping portfolio）
    """
    # 只在rebalance日有信号（每隔holding_period天换仓）
    signal = signal.with_columns(
        (pl.col("date").rank("dense").over("factset_entity_id") % config.holding_period == 0)
        .alias("is_rebalance_day")
    )

    if config.long_short:
        n = config.n_quantiles
        signal = signal.with_columns(
            pl.when(pl.col("quantile_lagged") == n - 1).then(1.0)   # top quantile: long
             .when(pl.col("quantile_lagged") == 0).then(-1.0)        # bottom quantile: short
             .otherwise(0.0)
             .alias("raw_weight")
        )
    else:
        # 纯多头：用zscore正比分配权重
        signal = signal.with_columns(
            pl.col("signal_lagged").clip(0, None).alias("raw_weight")
        )

    # 截面归一化权重（多头+空头分别归一）
    signal = signal.with_columns(
        (pl.col("raw_weight") / pl.col("raw_weight").abs().sum().over("date"))
        .alias("weight")
    )

    return signal

# compte retu
def compute_returns(
    positions: pl.DataFrame,
    returns: pl.DataFrame,
    config: BacktestConfig
) -> pl.DataFrame:
    """
    组合收益 = sum(weight_i * ret_i) - transaction_cost
    """
    # join收益率
    pnl = positions.join(
        returns.select(["factset_entity_id", "date", "ret"]),
        on=["factset_entity_id", "date"],
        how="inner"
    )

    # 计算每日换手率
    pnl = pnl.sort(["factset_entity_id", "date"]).with_columns(
        (pl.col("weight") - pl.col("weight").shift(1).over("factset_entity_id"))
        .abs()
        .alias("turnover")
    )

    # 组合日收益
    daily_pnl = (
        pnl
        .with_columns(
            (pl.col("weight") * pl.col("ret")).alias("gross_pnl"),
            (pl.col("turnover") * config.transaction_cost).alias("cost"),
        )
        .group_by("date")
        .agg(
            pl.col("gross_pnl").sum().alias("gross_ret"),
            pl.col("cost").sum().alias("total_cost"),
        )
        .with_columns(
            (pl.col("gross_ret") - pl.col("total_cost")).alias("net_ret")
        )
        .sort("date")
    )

    return daily_pnl

# IC
def compute_ic(positions: pl.DataFrame, returns: pl.DataFrame) -> pl.DataFrame:
    """
    每日 IC = corr(signal, forward_ret)
    """
    merged = positions.join(
        returns.select(["factset_entity_id", "date", "ret"]),
        on=["factset_entity_id", "date"],
        how="inner"
    )

    ic = (
        merged
        .group_by("date")
        .agg(
            pl.pearsonr("signal_lagged", "ret").alias("IC"),
            pl.spearmanr("signal_lagged", "ret").alias("RankIC"),  # 如果polars版本支持
        )
        .sort("date")
    )
    return ic


def compute_quantile_returns(positions: pl.DataFrame, returns: pl.DataFrame) -> pl.DataFrame:
    """
    每个分位组的平均日收益
    """
    merged = positions.join(
        returns.select(["factset_entity_id", "date", "ret"]),
        on=["factset_entity_id", "date"],
        how="inner"
    )

    quantile_ret = (
        merged
        .group_by(["date", "quantile_lagged"])
        .agg(pl.col("ret").mean().alias("avg_ret"))
        .sort(["date", "quantile_lagged"])
    )
    return quantile_ret

# stat
def performance_stats(daily_pnl: pl.DataFrame, ic_df: pl.DataFrame) -> dict:
    rets = daily_pnl["net_ret"].to_numpy()
    gross = daily_pnl["gross_ret"].to_numpy()
    ic = ic_df["IC"].drop_nulls().to_numpy()

    ann = 252
    stats = {
        # 收益
        "Ann Return":       rets.mean() * ann,
        "Ann Volatility":   rets.std() * np.sqrt(ann),
        "Sharpe":           rets.mean() / (rets.std() + 1e-10) * np.sqrt(ann),
        # 回撤
        "Max Drawdown":     max_drawdown(rets),
        # 成本
        "Ann Turnover":     daily_pnl["total_cost"].mean() / 0.001 * ann,  # 换手率（单边）
        "Ann Cost":         daily_pnl["total_cost"].mean() * ann,
        # IC
        "IC Mean":          ic.mean(),
        "IC Std":           ic.std(),
        "ICIR":             ic.mean() / (ic.std() + 1e-10),
        "IC > 0 %":         (ic > 0).mean(),
    }
    return stats


def max_drawdown(rets: np.ndarray) -> float:
    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return dd.min()

# overall
def run_backtest(
    etf_signal: pl.DataFrame,
    returns: pl.DataFrame,
    config: BacktestConfig = BacktestConfig()
) -> dict:

    signal   = prepare_signal(etf_signal, config)
    positions = build_positions(signal, config)
    daily_pnl = compute_returns(positions, returns, config)
    ic_df     = compute_ic(positions, returns)
    q_rets    = compute_quantile_returns(positions, returns)
    stats     = performance_stats(daily_pnl, ic_df)

    print("\n====== Backtest Results ======")
    for k, v in stats.items():
        print(f"  {k:<20}: {v:.4f}")

    return {
        "stats": stats,
        "daily_pnl": daily_pnl,
        "ic": ic_df,
        "quantile_returns": q_rets,
        "positions": positions,
    }

# 运行
results = run_backtest(etf_signal, etf_returns)
