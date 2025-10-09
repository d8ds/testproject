import polars as pl

def compute_complete_signal_events(filing_df: pl.DataFrame, 
                                   lookback_days: int = 180) -> pl.DataFrame:
    """
    高效计算完整的signal变化事件
    """
    
    filing_df = filing_df.with_columns(pl.col("date").cast(pl.Date)).sort(["qid", "date"])
    
    # 为每个filing创建两个事件：进入和退出
    events = pl.concat([
        # 事件类型1：filing发布（进入窗口）
        filing_df.select([
            "qid",
            pl.col("date").alias("event_date"),
            pl.col("date").alias("filing_date"),
            pl.col("length"),
            pl.lit("enter").alias("event_type")
        ]),
        
        # 事件类型2：filing过期（退出窗口）
        filing_df.select([
            "qid", 
            (pl.col("date") + pl.duration(days=lookback_days + 1)).alias("event_date"),
            pl.col("date").alias("filing_date"),
            pl.col("length"),
            pl.lit("exit").alias("event_type")
        ])
    ]).sort(["qid", "event_date", "event_type"])
    
    # 对每个事件，计算窗口内有效的filings
    results = []
    
    for qid in events["qid"].unique().to_list():
        company_events = events.filter(pl.col("qid") == qid).sort("event_date")
        company_filings = filing_df.filter(pl.col("qid") == qid).sort("date")
        
        active_filings = []  # 当前窗口内的filing列表
        prev_signal = None
        
        for row in company_events.iter_rows(named=True):
            event_date = row["event_date"]
            event_type = row["event_type"]
            filing_date = row["filing_date"]
            length = row["length"]
            
            if event_type == "enter":
                # Filing进入窗口
                active_filings.append((filing_date, length))
                active_filings.sort(key=lambda x: x[0], reverse=True)  # 按日期降序
                
            else:  # exit
                # Filing退出窗口
                active_filings = [(d, l) for d, l in active_filings if d != filing_date]
            
            # 计算当前signal
            if len(active_filings) >= 2:
                current_signal = active_filings[0][1] - active_filings[1][1]
            else:
                current_signal = None
            
            # 只记录signal变化
            if current_signal != prev_signal:
                results.append({
                    "qid": qid,
                    "date": event_date,
                    "signal": current_signal,
                    "n_filings_in_window": len(active_filings)
                })
                prev_signal = current_signal
    
    return pl.DataFrame(results).sort(["qid", "date"])


# 使用示例
signal_events = compute_complete_signal_events(df, lookback_days=180)

# 保存signal事件
signal_events.write_parquet("complete_signal_events.parquet")

# 在backtest中使用
def get_signals_for_backtest(signal_events: pl.DataFrame, 
                             backtest_dates: pl.DataFrame) -> pl.DataFrame:
    """
    backtest_dates: DataFrame with [qid, date]
    """
    return backtest_dates.join_asof(
        signal_events.select(["qid", "date", "signal"]),
        on="date",
        by="qid",
        strategy="backward"
    )
