import polars as pl

PORTFOLIO_NAV   = 1_000_000
WEIGHT_CAP      = 0.30
REBAL_THRESHOLD = 0.02
METHOD          = "zscore"  # "topn" | "zscore" | "rank_linear"
TOP_N           = 5         # 仅 method="topn" 时生效

# ── Step 1: 信号 → 截面权重 ───────────────────────────
def signal_to_weight(df: pl.DataFrame) -> pl.DataFrame:
    if METHOD == "topn":
        df = (
            df.with_columns(
                pl.col("signal_value").rank("dense", descending=True).over("date").alias("rank")
            )
            .filter(pl.col("rank") <= TOP_N)
            .with_columns((1.0 / pl.lit(TOP_N)).alias("raw_weight"))
        )
    elif METHOD == "zscore":
        df = (
            df.with_columns([
                pl.col("signal_value").mean().over("date").alias("mu"),
                pl.col("signal_value").std().over("date").alias("sigma"),
            ])
            .with_columns(
                ((pl.col("signal_value") - pl.col("mu")) / pl.col("sigma"))
                .clip(lower_bound=0)
                .alias("raw_weight")
            )
        )
    elif METHOD == "rank_linear":
        df = df.with_columns(
            (1.0 / pl.col("signal_value").rank(descending=True).over("date")).alias("raw_weight")
        )

    return (
        df
        .with_columns(
            (pl.col("raw_weight") / pl.col("raw_weight").sum().over("date")).alias("weight")
        )
        .with_columns(pl.col("weight").clip(upper_bound=WEIGHT_CAP))
        .with_columns(
            (pl.col("weight") / pl.col("weight").sum().over("date")).alias("weight")
        )
        .select(["date", "sector_id", "weight"])
    )

# ── Step 2: 换手缓冲 ──────────────────────────────────
def apply_rebalance_buffer(target: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    return (
        target
        .join(prev.rename({"weight": "prev_weight"}), on="sector_id", how="left")
        .with_columns(pl.col("prev_weight").fill_null(0.0))
        .with_columns(
            pl.when((pl.col("weight") - pl.col("prev_weight")).abs() > REBAL_THRESHOLD)
              .then(pl.col("weight"))
              .otherwise(pl.col("prev_weight"))
              .alias("final_weight")
        )
        .with_columns(
            (pl.col("final_weight") / pl.col("final_weight").sum()).alias("final_weight")
        )
        .select(["date", "sector_id", "final_weight"])
    )

# ── Step 3: 权重 → 持仓股数 ───────────────────────────
def weight_to_position(weights: pl.DataFrame, price_df: pl.DataFrame) -> pl.DataFrame:
    return (
        weights
        .join(price_df.select(["sector_id", "date", "close"]), on=["sector_id", "date"], how="left")
        .with_columns((pl.col("final_weight") * PORTFOLIO_NAV).alias("notional"))
        .with_columns((pl.col("notional") / pl.col("close")).floor().alias("shares"))
        .with_columns((pl.col("shares") * pl.col("close")).alias("actual_notional"))
    )

# ── Step 4: 生成交易指令 ──────────────────────────────
def generate_trades(prev_pos: pl.DataFrame, curr_pos: pl.DataFrame) -> pl.DataFrame:
    return (
        curr_pos.select(["date", "sector_id", "shares"])
        .join(prev_pos.select(["sector_id", "shares"]).rename({"shares": "prev_shares"}),
              on="sector_id", how="left")
        .with_columns(pl.col("prev_shares").fill_null(0))
        .with_columns((pl.col("shares") - pl.col("prev_shares")).alias("trade_shares"))
        .with_columns(
            pl.when(pl.col("trade_shares") > 0).then(pl.lit("BUY"))
              .when(pl.col("trade_shares") < 0).then(pl.lit("SELL"))
              .otherwise(pl.lit("HOLD"))
              .alias("side")
        )
        .filter(pl.col("trade_shares") != 0)
    )

# ── 主流程 ────────────────────────────────────────────
# signal_df 列：sector_id, date, signal_value
signal_df = signal_df.with_columns(
    pl.col("signal_value").shift(1).over("sector_id")  # 防前视偏差
).drop_nulls()

rebal_dates = signal_df["date"].unique().sort()
all_positions = []
prev_weights  = pl.DataFrame({"sector_id": [], "weight": []}, schema={"sector_id": pl.Utf8, "weight": pl.Float64})
prev_pos      = pl.DataFrame({"sector_id": [], "shares": []}, schema={"sector_id": pl.Utf8, "shares": pl.Float64})

for date in rebal_dates:
    day_signal = signal_df.filter(pl.col("date") == date)
    weights    = signal_to_weight(day_signal)
    final_w    = apply_rebalance_buffer(weights, prev_weights)
    pos        = weight_to_position(final_w, price_df)
    trades     = generate_trades(prev_pos, pos)

    all_positions.append(pos)
    prev_weights = final_w.select(["sector_id", "final_weight"]).rename({"final_weight": "weight"})
    prev_pos     = pos.select(["sector_id", "shares"])

positions = pl.concat(all_positions)
# signal_df → position_df，无需价格
signal_shifted = signal_df.with_columns(
    pl.col("signal_value").shift(1).over("sector_id")
).drop_nulls()

weights = signal_to_weight(signal_shifted)          # Step 1
final_w = apply_rebalance_buffer(weights, prev_w)   # Step 2（可选）

# final_w 就是 holding position
# 列：date, sector_id, final_weight

