import polars as pl
from datetime import datetime, timedelta
from typing import Optional

class SectorSignalBuilder:
    """
    处理成分股变化的Sector Level信号构建器
    """
    
    def __init__(
        self,
        lookback_days: int = 30,
        min_companies: int = 3,
        min_weight_coverage: float = 0.2,
        signal_decay_halflife: int = 7,
        gap_threshold_days: int = 60
    ):
        """
        参数:
        - lookback_days: rolling window长度
        - min_companies: 最小公司数要求
        - min_weight_coverage: 最小权重覆盖率
        - signal_decay_halflife: EMA decay半衰期
        - gap_threshold_days: 判断公司退出的时间间隔阈值
        """
        self.lookback_days = lookback_days
        self.min_companies = min_companies
        self.min_weight_coverage = min_weight_coverage
        self.signal_decay_halflife = signal_decay_halflife
        self.gap_threshold_days = gap_threshold_days
    
    def convert_to_interval_format(
        self, 
        df_sector: pl.DataFrame
    ) -> pl.DataFrame:
        """
        将 (sectorId, qid, date, weight) 转换为 (sectorId, qid, startDate, endDate, weight)
        处理权重变化和公司进出sector
        
        参数:
        - df_sector: 包含列 [sectorId, qid, date, weight]
        
        返回:
        - 包含列 [sectorId, qid, startDate, endDate, weight, segment_id]
        """
        
        # 先过滤掉无效数据
        df_clean = (
            df_sector
            .filter(
                pl.col('weight').is_not_null() & 
                (pl.col('weight') > 0)
            )
            .sort(['sectorId', 'qid', 'date'])
        )
        
        interval_df = (
            df_clean
            .with_columns([
                # 获取上一条记录的date和weight
                pl.col('date')
                  .shift(1)
                  .over(['sectorId', 'qid'])
                  .alias('prev_date'),
                pl.col('weight')
                  .shift(1)
                  .over(['sectorId', 'qid'])
                  .alias('prev_weight')
            ])
            .with_columns([
                # 定义"新段"的条件
                (
                    # 第一条记录
                    pl.col('prev_weight').is_null() |
                    # 或weight变化（容忍小的浮点误差）
                    ((pl.col('weight') - pl.col('prev_weight')).abs() > 1e-6) |
                    # 或时间间断（公司退出后重新加入）
                    ((pl.col('date') - pl.col('prev_date')).dt.total_days() > self.gap_threshold_days)
                ).alias('is_new_segment')
            ])
            .with_columns([
                # 为每个segment分配ID
                pl.col('is_new_segment')
                  .cum_sum()
                  .over(['sectorId', 'qid'])
                  .alias('segment_id')
            ])
            # 聚合每个segment
            .group_by(['sectorId', 'qid', 'segment_id', 'weight'])
            .agg([
                pl.col('date').min().alias('startDate'),
                pl.col('date').max().alias('endDate'),
                pl.count().alias('n_dates')
            ])
            .sort(['sectorId', 'qid', 'startDate'])
        )
        
        # 验证：确保同一公司的segments不重叠
        validation = self._validate_intervals(interval_df)
        if len(validation) > 0:
            print("⚠️  Warning: Found overlapping intervals:")
            print(validation)
        
        return interval_df
    
    def _validate_intervals(self, interval_df: pl.DataFrame) -> pl.DataFrame:
        """验证区间是否重叠"""
        return (
            interval_df
            .sort(['sectorId', 'qid', 'startDate'])
            .with_columns([
                pl.col('startDate')
                  .shift(-1)
                  .over(['sectorId', 'qid'])
                  .alias('next_startDate')
            ])
            .filter(
                pl.col('next_startDate').is_not_null() &
                (pl.col('endDate') >= pl.col('next_startDate'))
            )
        )
    
    def create_sector_signal(
        self,
        df_company: pl.DataFrame,  # [id, documentid, date, sentiment]
        df_sector: pl.DataFrame     # [sectorId, qid, date, weight]
    ) -> pl.DataFrame:
        """
        创建sector level信号，处理成分股变化和稀疏性
        
        返回:
        - 包含列 [sectorId, date, sector_sentiment, n_companies, weight_coverage, 
                  signal_quality, final_signal]
        """
        
        # Step 1: 转换为interval格式
        print("Step 1: Converting to interval format...")
        df_sector_interval = self.convert_to_interval_format(df_sector)
        print(f"  Created {len(df_sector_interval)} intervals")
        
        # Step 2: 为每个company sentiment找到对应的point-in-time weight
        print("\nStep 2: Joining sentiment with point-in-time weights...")
        sentiment_with_pit_weight = self._join_pit_weights(
            df_company, 
            df_sector_interval
        )
        print(f"  Matched {len(sentiment_with_pit_weight)} sentiment records")
        
        # Step 3: 创建rolling aggregation
        print("\nStep 3: Creating rolling sector signals...")
        sector_signal = self._create_rolling_sector_signal(
            sentiment_with_pit_weight
        )
        
        # Step 4: 应用coverage filter和信号平滑
        print("\nStep 4: Applying coverage filters and signal smoothing...")
        final_signal = self._apply_signal_filters(sector_signal)
        
        print(f"\n✓ Created signals for {final_signal['sectorId'].n_unique()} sectors")
        print(f"  Date range: {final_signal['date'].min()} to {final_signal['date'].max()}")
        
        return final_signal
    
    def _join_pit_weights(
        self,
        df_company: pl.DataFrame,
        df_sector_interval: pl.DataFrame
    ) -> pl.DataFrame:
        """
        使用interval join找到每个sentiment对应的point-in-time weight
        """
        
        # 确保date列是Date类型
        df_company = df_company.with_columns(pl.col('date').cast(pl.Date))
        df_sector_interval = df_sector_interval.with_columns([
            pl.col('startDate').cast(pl.Date),
            pl.col('endDate').cast(pl.Date)
        ])
        
        # Interval join: 找到sentiment date在哪个interval内
        sentiment_with_weight = (
            df_company
            .join(
                df_sector_interval,
                how='inner',
                left_on='id',
                right_on='qid'
            )
            .filter(
                (pl.col('date') >= pl.col('startDate')) &
                (pl.col('date') <= pl.col('endDate'))
            )
            .select([
                'sectorId',
                'id',
                'documentid',
                'date',
                'sentiment',
                'weight',
                'segment_id'
            ])
        )
        
        return sentiment_with_weight
    
    def _create_rolling_sector_signal(
        self,
        sentiment_with_weight: pl.DataFrame
    ) -> pl.DataFrame:
        """
        创建rolling window sector signal
        """
        
        # 计算weighted sentiment
        df_weighted = sentiment_with_weight.with_columns([
            (pl.col('sentiment') * pl.col('weight')).alias('weighted_sentiment')
        ])
        
        # 使用group_by_dynamic进行rolling aggregation
        sector_signal = (
            df_weighted
            .sort('date')
            .group_by_dynamic(
                'date',
                every='1d',
                period=f'{self.lookback_days}d',
                by='sectorId'
            )
            .agg([
                # 聚合weighted sentiment
                pl.col('weighted_sentiment').sum().alias('sum_weighted_sentiment'),
                pl.col('weight').sum().alias('sum_weights'),
                
                # 统计coverage指标
                pl.col('id').n_unique().alias('n_companies'),
                pl.col('documentid').count().alias('n_docs'),
                
                # 额外的质量指标
                pl.col('sentiment').std().alias('sentiment_std'),
                pl.col('weight').max().alias('max_single_weight')
            ])
            .with_columns([
                # 归一化：sector_sentiment
                pl.when(pl.col('sum_weights') > 0)
                  .then(pl.col('sum_weighted_sentiment') / pl.col('sum_weights'))
                  .otherwise(None)
                  .alias('sector_sentiment'),
                
                # 权重覆盖率
                pl.col('sum_weights').alias('weight_coverage')
            ])
        )
        
        return sector_signal
    
    def _apply_signal_filters(
        self,
        sector_signal: pl.DataFrame
    ) -> pl.DataFrame:
        """
        应用coverage filter、信号平滑和quality控制
        """
        
        filtered_signal = (
            sector_signal
            .sort(['sectorId', 'date'])
            .with_columns([
                # 计算信号质量分数
                self._calculate_signal_quality().alias('signal_quality'),
                
                # 只有满足条件才接受新信号
                pl.when(
                    (pl.col('n_companies') >= self.min_companies) &
                    (pl.col('weight_coverage') >= self.min_weight_coverage) &
                    (pl.col('sentiment_std').is_not_null())  # 确保有variance
                )
                .then(pl.col('sector_sentiment'))
                .otherwise(None)
                .alias('raw_signal')
            ])
            .with_columns([
                # EMA smoothing - 降低turnover
                pl.col('raw_signal')
                  .ewm_mean(
                      half_life=f'{self.signal_decay_halflife}d',
                      by='date'
                  )
                  .over('sectorId')
                  .alias('smoothed_signal')
            ])
            .with_columns([
                # 如果当前没有新信号，forward fill上一个good signal
                pl.col('smoothed_signal')
                  .forward_fill()
                  .over('sectorId')
                  .alias('final_signal')
            ])
        )
        
        return filtered_signal
    
    def _calculate_signal_quality(self) -> pl.Expr:
        """
        计算信号质量分数 (0-1)
        考虑: coverage, diversity, concentration
        """
        return (
            # Coverage分数
            pl.min_horizontal([
                pl.col('n_companies') / (self.min_companies * 2),
                pl.lit(1.0)
            ]) * 0.4 +
            
            # Weight coverage分数
            pl.min_horizontal([
                pl.col('weight_coverage') / 0.5,
                pl.lit(1.0)
            ]) * 0.3 +
            
            # Diversity分数 (避免过度集中)
            (1.0 - pl.col('max_single_weight')) * 0.3
        )
    
    def generate_signal_statistics(
        self,
        final_signal: pl.DataFrame
    ) -> pl.DataFrame:
        """
        生成信号质量统计报告
        """
        
        stats = (
            final_signal
            .group_by('sectorId')
            .agg([
                # 基本统计
                pl.count().alias('n_dates'),
                pl.col('final_signal').is_not_null().sum().alias('n_valid_signals'),
                
                # 信号质量
                pl.col('signal_quality').mean().alias('avg_quality'),
                pl.col('n_companies').mean().alias('avg_coverage'),
                pl.col('weight_coverage').mean().alias('avg_weight_cov'),
                
                # 信号特性
                pl.col('final_signal').std().alias('signal_volatility'),
                pl.col('final_signal').diff().abs().mean().alias('avg_turnover'),
                
                # 稀疏性
                (pl.col('n_companies') < self.min_companies).sum().alias('n_low_coverage_days'),
                pl.col('max_single_weight').mean().alias('avg_concentration')
            ])
            .with_columns([
                (pl.col('n_valid_signals') / pl.col('n_dates')).alias('signal_coverage_pct')
            ])
            .sort('sectorId')
        )
        
        return stats


# ============== 辅助函数 ==============

def create_sample_data():
    """创建测试数据"""
    
    # Company level sentiment data
    df_company = pl.DataFrame({
        'id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
        'documentid': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9'],
        'date': [
            '2024-01-05', '2024-01-15', '2024-02-10',
            '2024-01-08', '2024-02-05',
            '2024-01-20', '2024-02-15',
            '2024-01-25', '2024-02-20'
        ],
        'sentiment': [0.5, 0.6, 0.4, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9]
    }).with_columns(pl.col('date').str.to_date())
    
    # Sector composition with rebalancing
    # Sector 1在2024-02-01发生rebalancing，权重变化
    df_sector = pl.DataFrame({
        'sectorId': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
        'qid': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'C', 'D'],
        'date': [
            '2024-01-01', '2024-02-01',  # A的权重变化
            '2024-01-01', '2024-02-01',  # B的权重变化
            '2024-01-01', '2024-02-01',  # C的权重变化
            '2024-01-01', '2024-02-01',  # D加入sector
            '2024-01-01', '2024-01-01'   # Sector 2
        ],
        'weight': [0.4, 0.3, 0.4, 0.3, 0.2, 0.2, 0.0, 0.2, 0.6, 0.4]
    }).with_columns(pl.col('date').str.to_date())
    
    return df_company, df_sector


# ============== 主函数 ==============

def main():
    """完整示例"""
    
    print("=" * 60)
    print("Sector Signal Builder - Complete Example")
    print("=" * 60)
    
    # 1. 创建测试数据（或加载你的真实数据）
    df_company, df_sector = create_sample_data()
    
    print("\nInput Data:")
    print(f"  Company sentiment records: {len(df_company)}")
    print(f"  Sector composition records: {len(df_sector)}")
    
    # 2. 初始化signal builder
    builder = SectorSignalBuilder(
        lookback_days=30,
        min_companies=2,  # 测试数据较小，降低阈值
        min_weight_coverage=0.15,
        signal_decay_halflife=7,
        gap_threshold_days=60
    )
    
    # 3. 创建sector signal
    print("\n" + "=" * 60)
    final_signal = builder.create_sector_signal(df_company, df_sector)
    
    # 4. 查看结果
    print("\n" + "=" * 60)
    print("Final Signals (sample):")
    print("=" * 60)
    print(
        final_signal
        .filter(pl.col('final_signal').is_not_null())
        .head(10)
    )
    
    # 5. 生成统计报告
    print("\n" + "=" * 60)
    print("Signal Quality Statistics:")
    print("=" * 60)
    stats = builder.generate_signal_statistics(final_signal)
    print(stats)
    
    # 6. 保存结果
    # final_signal.write_parquet('sector_signals.parquet')
    # stats.write_csv('signal_statistics.csv')
    
    return final_signal, stats


if __name__ == '__main__':
    final_signal, stats = main()
