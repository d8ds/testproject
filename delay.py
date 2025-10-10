class FilingSignalV2:
    """
    不依赖行业的增强signal系统
    """
    
    def __init__(self, filing_df: pl.DataFrame):
        self.df = filing_df.with_columns([
            pl.col("date").cast(pl.Date),
            (pl.col("filing_date") - pl.col("event_date")).dt.total_seconds().alias("delay_seconds"),
            (pl.col("filing_date") - pl.col("event_date")).dt.total_days().alias("delay_days"),
        ]).sort(["qid", "date"])
    
    def generate_base_signals(self) -> pl.DataFrame:
        """
        生成基础信号（不需要行业）
        """
        
        df = self.df
        
        # ===== 1. 个股历史标准化 =====
        df = df.with_columns([
            # 历史均值和标准差（过去所有filing）
            pl.col("delay_seconds").mean().over("qid").alias("qid_mean_delay"),
            pl.col("delay_seconds").std().over("qid").alias("qid_std_delay"),
            
            # 滚动均值（最近4次filing）
            pl.col("delay_seconds").rolling_mean(4).over("qid").alias("qid_ma4_delay"),
            pl.col("delay_seconds").rolling_std(4).over("qid").alias("qid_std4_delay"),
            
            # 前一次delay
            pl.col("delay_seconds").shift(1).over("qid").alias("prev_delay"),
        ])
        
        # Z-scores
        df = df.with_columns([
            # 相对于全历史
            ((pl.col("delay_seconds") - pl.col("qid_mean_delay")) / 
             pl.col("qid_std_delay")).alias("delay_z_hist"),
            
            # 相对于近期（更敏感）
            ((pl.col("delay_seconds") - pl.col("qid_ma4_delay")) / 
             pl.col("qid_std4_delay")).alias("delay_z_recent"),
            
            # 变化率
            ((pl.col("delay_seconds") - pl.col("prev_delay")) / 
             pl.col("prev_delay")).alias("delay_pct_change"),
        ])
        
        # ===== 2. 截面标准化（每个日期横截面） =====
        df = df.with_columns([
            # Rank标准化 (0-1)
            pl.col("delay_seconds").rank().over("date").alias("delay_rank_raw"),
            pl.col("qid").count().over("date").alias("n_stocks"),
        ]).with_columns([
            (pl.col("delay_rank_raw") / pl.col("n_stocks")).alias("delay_rank_pct"),
            
            # 截面Z-score
            ((pl.col("delay_seconds") - pl.col("delay_seconds").mean().over("date")) /
             pl.col("delay_seconds").std().over("date")).alias("delay_z_cross"),
        ])
        
        # ===== 3. 趋势和加速度 =====
        df = df.with_columns([
            # 延迟趋势（是在恶化还是改善）
            (pl.col("delay_seconds") - pl.col("delay_seconds").shift(1).over("qid")
            ).alias("delay_delta1"),
            
            (pl.col("delay_seconds").shift(1).over("qid") - 
             pl.col("delay_seconds").shift(2).over("qid")
            ).alias("delay_delta2"),
        ]).with_columns([
            # 加速度：变化的变化
            (pl.col("delay_delta1") - pl.col("delay_delta2")).alias("delay_acceleration"),
        ])
        
        # ===== 4. Length相关特征 =====
        df = df.with_columns([
            pl.col("length").shift(1).over("qid").alias("prev_length"),
            pl.col("length").rolling_mean(4).over("qid").alias("ma4_length"),
        ]).with_columns([
            (pl.col("length") - pl.col("prev_length")).alias("length_change"),
            ((pl.col("length") - pl.col("ma4_length")) / pl.col("ma4_length")).alias("length_surprise"),
        ])
        
        # ===== 5. 组合信号 =====
        df = df.with_columns([
            # Signal 1: 极端delay（个股视角）
            pl.when(pl.col("delay_z_recent").abs() > 2)
              .then(-pl.col("delay_z_recent"))  # 负号：delay大 = 负信号
              .otherwise(0)
              .alias("signal_extreme_delay"),
            
            # Signal 2: 加速恶化
            pl.when(
                (pl.col("delay_z_recent") > 1) &  # delay已经很大
                (pl.col("delay_acceleration") > 0)  # 还在恶化
            ).then(-2.0)  # 强负信号
              .when(
                (pl.col("delay_z_recent") < -1) &  # delay很小
                (pl.col("delay_acceleration") < 0)  # 还在改善
            ).then(2.0)  # 强正信号
              .otherwise(0)
              .alias("signal_accelerating"),
            
            # Signal 3: 截面极端值
            pl.when(pl.col("delay_rank_pct") < 0.05)  # 最快5%
              .then(1.5)
              .when(pl.col("delay_rank_pct") > 0.95)  # 最慢5%
              .then(-1.5)
              .otherwise(0)
              .alias("signal_cross_extreme"),
            
            # Signal 4: Delay + Length组合
            # 当delay大且length也在减少时（公司可能有问题）
            pl.when(
                (pl.col("delay_z_recent") > 1.5) &
                (pl.col("length_change") < 0)
            ).then(-1.5)
              .when(
                (pl.col("delay_z_recent") < -1.5) &
                (pl.col("length_change") > 0)
            ).then(1.5)
              .otherwise(0)
              .alias("signal_delay_length"),
            
            # Signal 5: 综合加权
            (
                0.3 * pl.col("signal_extreme_delay").fill_null(0) +
                0.3 * pl.col("signal_accelerating").fill_null(0) +
                0.2 * pl.col("signal_cross_extreme").fill_null(0) +
                0.2 * pl.col("signal_delay_length").fill_null(0)
            ).alias("signal_composite"),
        ])
        
        return df
    
    def create_quintile_portfolios(self, signal_col: str = "signal_composite") -> pl.DataFrame:
        """
        创建五分位组合（多空 + 中间组）
        """
        
        df = self.generate_base_signals()
        
        # 每个日期分成5组
        portfolios = df.with_columns([
            pl.col(signal_col).qcut(5, labels=["Q1_Short", "Q2", "Q3", "Q4", "Q5_Long"])
              .over("date")
              .alias("quintile")
        ])
        
        return portfolios
    
    def backtest_signal(self, 
                       signal_col: str,
                       returns_df: pl.DataFrame,
                       holding_period: int = 20) -> dict:
        """
        简单回测框架
        
        Parameters:
        -----------
        signal_col : 要测试的信号列
        returns_df : 包含 qid, date, forward_return_Nd 的收益数据
        holding_period : 持有期（天）
        """
        
        signals = self.generate_base_signals()
        
        # 合并收益数据
        backtest_df = signals.join(
            returns_df.select(["qid", "date", f"forward_return_{holding_period}d"]),
            on=["qid", "date"],
            how="inner"
        )
        
        # 计算每个日期的多空组合收益
        portfolio_returns = backtest_df.with_columns([
            # 排序
            pl.col(signal_col).rank().over("date").alias("signal_rank"),
            pl.col("qid").count().over("date").alias("n_stocks_date"),
        ]).with_columns([
            # Top 20% = Long, Bottom 20% = Short
            pl.when(pl.col("signal_rank") > pl.col("n_stocks_date") * 0.8)
              .then(pl.col(f"forward_return_{holding_period}d"))
              .when(pl.col("signal_rank") <= pl.col("n_stocks_date") * 0.2)
              .then(-pl.col(f"forward_return_{holding_period}d"))
              .otherwise(None)
              .alias("portfolio_return")
        ]).filter(pl.col("portfolio_return").is_not_null())
        
        # 按日期汇总
        daily_returns = portfolio_returns.group_by("date").agg([
            pl.col("portfolio_return").mean().alias("daily_return")
        ]).sort("date")
        
        # 计算统计指标
        mean_return = daily_returns["daily_return"].mean()
        std_return = daily_returns["daily_return"].std()
        sharpe = mean_return / std_return * (252 ** 0.5)  # 年化
        
        return {
            "sharpe": sharpe,
            "mean_return": mean_return,
            "std_return": std_return,
            "n_periods": len(daily_returns)
        }


# ============= 完整使用示例 =============

# 1. 创建signal生成器
signal_gen = FilingSignalV2(filing_df)

# 2. 生成所有信号
signals = signal_gen.generate_base_signals()

# 3. 测试不同信号
test_signals = [
    "signal_extreme_delay",
    "signal_accelerating", 
    "signal_cross_extreme",
    "signal_delay_length",
    "signal_composite"
]

print("Signal Performance:")
print("-" * 60)
for sig in test_signals:
    stats = signal_gen.backtest_signal(sig, returns_df, holding_period=20)
    print(f"{sig:30s} | Sharpe: {stats['sharpe']:.2f} | "
          f"Mean: {stats['mean_return']*10000:.1f}bps | "
          f"Std: {stats['std_return']*10000:.1f}bps")

# 4. 创建最终组合
final_portfolio = signal_gen.create_quintile_portfolios(signal_col="signal_composite")
