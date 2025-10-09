import polars as pl
from datetime import timedelta

class FilingSignalCalculatorComplete:
    """
    完整版本：考虑filing进入和退出6个月窗口
    """
    
    def __init__(self, filing_df: pl.DataFrame, lookback_days: int = 180):
        self.filing_df = filing_df.with_columns(
            pl.col("date").cast(pl.Date)
        ).sort(["qid", "date"])
        self.lookback_days = lookback_days
        self.signal_events = None
    
    def compute_all_signal_events(self) -> pl.DataFrame:
        """
        计算所有导致signal变化的事件：
        1. 新filing发布（filing进入窗口）
        2. 旧filing超过6个月（filing退出窗口）
        """
        
        # === 事件1：新filing发布 ===
        filing_events = self.filing_df.with_columns([
            pl.col("date").alias("event_date"),
            pl.lit("filing").alias("event_type")
        ]).select(["qid", "event_date", "event_type"])
        
        # === 事件2：filing退出窗口（6个月后）===
        expiry_events = self.filing_df.with_columns([
            (pl.col("date") + pl.duration(days=self.lookback_days)).alias("event_date"),
            pl.lit("expiry").alias("event_type")
        ]).select(["qid", "event_date", "event_type"])
        
        # 合并所有事件并排序
        all_events = pl.concat([filing_events, expiry_events]).sort(["qid", "event_date"])
        
        # 为每个事件计算当前的signal
        signal_at_events = []
        
        for qid in all_events["qid"].unique().to_list():
            company_events = all_events.filter(pl.col("qid") == qid).sort("event_date")
            company_filings = self.filing_df.filter(pl.col("qid") == qid).sort("date")
            
            # 对每个事件日期计算signal
            event_signals = []
            for event_date in company_events["event_date"].to_list():
                signal = self._compute_signal_at_date(company_filings, event_date)
                event_signals.append({
                    "qid": qid,
                    "date": event_date,
                    "signal": signal
                })
            
            signal_at_events.extend(event_signals)
        
        result = pl.DataFrame(signal_at_events).sort(["qid", "date"])
        
        # 只保留signal发生变化的时点
        result = result.with_columns([
            pl.col("signal").shift(1).over("qid").alias("prev_signal")
        ]).filter(
            # 保留第一个事件或signal发生变化的事件
            (pl.col("prev_signal").is_null()) | 
            (pl.col("signal") != pl.col("prev_signal")) |
            (pl.col("signal").is_null() != pl.col("prev_signal").is_null())
        ).select(["qid", "date", "signal"])
        
        self.signal_events = result
        return result
    
    def _compute_signal_at_date(self, company_filings: pl.DataFrame, as_of_date) -> float:
        """
        计算某个日期的signal值
        找到该日期前6个月内最近的两次filing
        """
        # 找到as_of_date之前且在6个月窗口内的所有filings
        cutoff_date = as_of_date - timedelta(days=self.lookback_days)
        
        valid_filings = company_filings.filter(
            (pl.col("date") <= as_of_date) &
            (pl.col("date") > cutoff_date)
        ).sort("date", descending=True)
        
        if len(valid_filings) < 2:
            return None
        
        # 最近的两次filing
        latest = valid_filings[0, "length"]
        second_latest = valid_filings[1, "length"]
        
        return latest - second_latest
    
    def get_signal_at_dates(self, query_df: pl.DataFrame) -> pl.DataFrame:
        """
        获取指定日期的signal（使用前向填充）
        """
        if self.signal_events is None:
            self.compute_all_signal_events()
        
        query_df = query_df.with_columns(pl.col("date").cast(pl.Date))
        
        result = query_df.join_asof(
            self.signal_events,
            on="date",
            by="qid",
            strategy="backward"
        )
        
        return result


# ============= 使用示例 =============

# 创建测试数据来验证
test_data = pl.DataFrame({
    "qid": ["A"] * 5,
    "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
    "length": [100, 120, 110, 130, 125],
    "date": ["2023-01-01", "2023-03-01", "2023-05-01", "2023-07-01", "2023-09-01"]
})

calculator = FilingSignalCalculatorComplete(test_data, lookback_days=180)
signal_events = calculator.compute_all_signal_events()

print("Signal变化事件：")
print(signal_events)

# 查询任意日期的signal
query = pl.DataFrame({
    "qid": ["A"] * 10,
    "date": ["2023-01-15", "2023-03-15", "2023-05-15", "2023-07-15", 
             "2023-08-01", "2023-09-01", "2023-09-15", "2023-10-01",
             "2024-01-01", "2024-03-01"]
})

signals = calculator.get_signal_at_dates(query)
print("\n查询结果：")
print(signals)
