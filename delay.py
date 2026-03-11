# signal_df: (sector_id, date, signal_value)  ← 你的原始数据
# 注意：signal 必须 shift(1) 错位，避免前视偏差
signal_shifted = signal_df.with_columns(
    pl.col("signal_value").shift(1).over("sector_id")
).drop_nulls()

weights   = signal_to_weight(signal_shifted, method=METHOD)
final_w   = apply_rebalance_buffer(weights, prev_weights)   # 首期 prev_weights 为空
positions = weight_to_position(final_w, universe, price_df)
trades    = generate_trades(prev_positions, positions)

import polars as pl

# ── 参数 ──────────────────────────────────────────────
PORTFOLIO_NAV    = 1_000_000
TOP_N            = None        # None = 全部 sector，int = 只取前 N
WEIGHT_CAP       = 0.30        # 单 ETF 最大权重
REBAL_THRESHOLD  = 0.02        # 换手触发阈值（相对权重变化）
METHOD           = "zscore"    # "topn" | "zscore" | "rank_linear"

# ── Step 1: 信号 → 截面权重 ───────────────────────────
def signal_to_weight(signal_df: pl.DataFrame, method: str) -> pl.DataFrame:
    df = signal_df.sort(["date", "sector_id"])

    if method == "topn":
        df = (
            df.with_columns(
                pl.col("signal_value")
                  .rank("dense", descending=True)
                  .over("date")
                  .alias("rank")
            )
            .filter(pl.col("rank") <= TOP_N)
            .with_columns((1.0 / pl.col("rank").count().over("date")).alias("raw_weight"))
        )

    elif method == "zscore":
        df = (
            df.with_columns([
                pl.col("signal_value").mean().over("date").alias("mu"),
                pl.col("signal_value").std().over("date").alias("sigma"),
            ])
            .with_columns(
                ((pl.col("signal_value") - pl.col("mu")) / pl.col("sigma"))
                .clip(lower_bound=0)   # long-only：截掉负 z
                .alias("raw_weight")
            )
        )

    elif method == "rank_linear":
        df = (
            df.with_columns(
                pl.col("signal_value")
                  .rank(descending=True)
                  .over("date")
                  .alias("raw_rank")
            )
            .with_columns(
                (1.0 / pl.col("raw_rank")).alias("raw_weight")
            )
        )

    # 归一化 → 权重上限 → 再归一化
    df = (
        df.with_columns(
            (pl.col("raw_weight") / pl.col("raw_weight").sum().over("date"))
            .alias("weight")
        )
        .with_columns(
            pl.col("weight").clip(upper_bound=WEIGHT_CAP).alias("weight")
        )
        .with_columns(
            (pl.col("weight") / pl.col("weight").sum().over("date"))
            .alias("weight")
        )
    )
    return df.select(["date", "sector_id", "weight"])


# ── Step 2: 换手控制 ───────────────────────────────────
def apply_rebalance_buffer(
    target: pl.DataFrame,
    current: pl.DataFrame,   # (date, sector_id, weight) 上期持仓
) -> pl.DataFrame:
    """
    只有 |target_weight - current_weight| > threshold 时才真正调仓
    """
    df = (
        target.join(
            current.rename({"weight": "current_weight"}),
            on=["date", "sector_id"],
            how="left"
        )
        .with_columns(pl.col("current_weight").fill_null(0.0))
        .with_columns(
            (pl.col("weight") - pl.col("current_weight")).abs().alias("diff")
        )
        .with_columns(
            pl.when(pl.col("diff") > REBAL_THRESHOLD)
              .then(pl.col("weight"))
              .otherwise(pl.col("current_weight"))
              .alias("final_weight")
        )
        # 重新归一化（保证 long-only 总权重 = 1）
        .with_columns(
            (pl.col("final_weight") / pl.col("final_weight").sum().over("date"))
            .alias("final_weight")
        )
    )
    return df.select(["date", "sector_id", "final_weight"])


# ── Step 3: 权重 → ETF 持仓（股数）───────────────────
def weight_to_position(
    weight_df: pl.DataFrame,   # (date, sector_id, final_weight)
    universe:  pl.DataFrame,   # (sector_id, etf_ticker)
    price_df:  pl.DataFrame,   # (etf_ticker, date, close)
) -> pl.DataFrame:
    return (
        weight_df
        .join(universe, on="sector_id", how="left")
        .join(price_df, on=["etf_ticker", "date"], how="left")
        .with_columns([
            (pl.col("final_weight") * PORTFOLIO_NAV).alias("notional"),
        ])
        .with_columns(
            (pl.col("notional") / pl.col("close")).floor().alias("shares")
        )
        # 实际持仓金额（取整后）
        .with_columns(
            (pl.col("shares") * pl.col("close")).alias("actual_notional")
        )
        .select([
            "date", "sector_id", "etf_ticker",
            "final_weight", "notional", "shares", "actual_notional", "close"
        ])
    )


# ── Step 4: 生成交易指令（diff 前后持仓）────────────────
def generate_trades(
    prev_pos: pl.DataFrame,
    curr_pos: pl.DataFrame,
) -> pl.DataFrame:
    return (
        curr_pos.select(["date", "etf_ticker", "shares"])
        .join(
            prev_pos.select(["etf_ticker", "shares"]).rename({"shares": "prev_shares"}),
            on="etf_ticker",
            how="left"
        )
        .with_columns(pl.col("prev_shares").fill_null(0))
        .with_columns(
            (pl.col("shares") - pl.col("prev_shares")).alias("trade_shares")
        )
        .with_columns(
            pl.when(pl.col("trade_shares") > 0).then(pl.lit("BUY"))
             .when(pl.col("trade_shares") < 0).then(pl.lit("SELL"))
             .otherwise(pl.lit("HOLD"))
             .alias("side")
        )
        .filter(pl.col("trade_shares") != 0)
    )
