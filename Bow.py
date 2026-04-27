import polars as pl

def compute_rolling_uncertainty_signal(
    df: pl.DataFrame,
    window_days: int = 180,
    closed: str = "right",   # "right": 包含当天观测; "left": 排除当天
) -> pl.DataFrame:
    """
    输入:  df 含列 (qid, date, avg_uncertainty)
    输出:  每个 (qid, weekday) 一行,  signal = sum / count over [t - window_days, t]
    """
    df = df.with_columns(pl.col("date").cast(pl.Date))

    # 1. 构建 weekday 日历 (Mon-Fri)
    calendar = (
        pl.DataFrame({
            "date": pl.date_range(df["date"].min(), df["date"].max(),
                                  interval="1d", eager=True)
        })
        .filter(pl.col("date").dt.weekday() <= 5)   # 1=Mon ... 7=Sun
    )

    # 2. 笛卡尔积出 (qid, weekday) 全网格,  avg_uncertainty 置 null
    qids = df.select("qid").unique()
    grid = (
        qids.join(calendar, how="cross")
            .with_columns(
                pl.lit(None, dtype=df["avg_uncertainty"].dtype).alias("avg_uncertainty"),
                pl.lit(True).alias("_is_grid"),
            )
    )

    # 3. 与原始观测合并 (原始观测 _is_grid=False, 携带真实 avg_uncertainty)
    obs = df.with_columns(pl.lit(False).alias("_is_grid"))
    combined = pl.concat([obs, grid.select(obs.columns)]).sort(["qid", "date"])

    # 4. 每个 qid 内做时间窗口聚合.  rolling_*_by 自动按 date 找窗口边界,
    #    null 值不参与 sum/count, 因此 grid 行不会污染分子分母.
    period = f"{window_days}d"
    out = combined.with_columns([
        pl.col("avg_uncertainty")
          .rolling_sum_by("date", window_size=period, closed=closed)
          .over("qid").alias("signal_sum"),
        pl.col("avg_uncertainty").is_not_null().cast(pl.UInt32)
          .rolling_sum_by("date", window_size=period, closed=closed)
          .over("qid").alias("signal_count"),
    ]).with_columns(
        (pl.col("signal_sum") / pl.col("signal_count")).alias("signal_mean")
    )

    # 5. 只保留 grid 行 (= 每个 weekday 一条 live signal)
    return (
        out.filter(pl.col("_is_grid"))
           .select(["qid", "date", "signal_sum", "signal_count", "signal_mean"])
    )
