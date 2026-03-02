import pandas as pd
import numpy as np

def aggregate_with_asof_merge(broker_df, sector_df, max_lag_days=90):
    """
    使用as-of merge策略：为每个broker report找到最近的sector权重
    
    Parameters:
    -----------
    broker_df : DataFrame with [qid, date, document_id, sentiment]
    sector_df : DataFrame with [qid, date, weight, sector]
    max_lag_days : int, 最大允许的权重滞后天数
    
    Returns:
    --------
    sector_signals : DataFrame with sector-level signals
    diagnostics : dict with detailed statistics
    """
    
    # Step 1: 预处理
    broker_df = broker_df.copy()
    sector_df = sector_df.copy()
    
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    
    # 聚合document级别
    doc_sentiment = broker_df.groupby(
        ['qid', 'date', 'document_id']
    )['sentiment'].mean().reset_index()
    
    # 排序（merge_asof要求）
    doc_sentiment = doc_sentiment.sort_values(['qid', 'date'])
    sector_df = sector_df.sort_values(['qid', 'date'])
    
    print("开始merge_asof...")
    
    # Step 2: 使用merge_asof找最近的权重
    merged = pd.merge_asof(
        doc_sentiment,
        sector_df[['qid', 'date', 'weight', 'sector']],
        on='date',
        by='qid',
        direction='backward',  # 向前查找最近的权重
        tolerance=pd.Timedelta(days=max_lag_days)
    )
    
    # 记录权重日期用于诊断
    merged = merged.merge(
        sector_df[['qid', 'date', 'weight']],
        left_on=['qid', 'weight'],
        right_on=['qid', 'weight'],
        how='left',
        suffixes=('', '_weight')
    )
    
    merged['weight_lag_days'] = (merged['date'] - merged['date_weight']).dt.days
    
    print(f"Merge完成，匹配到权重的记录数: {merged['weight'].notna().sum()} / {len(merged)}")
    
    # Step 3: 过滤掉没有匹配到权重的记录
    merged_valid = merged[merged['weight'].notna()].copy()
    
    # Step 4: 按date-sector聚合
    # 重要：在聚合前需要重新标准化权重
    daily_sector_weight = merged_valid.groupby(['date', 'sector'])['weight'].sum().reset_index()
    daily_sector_weight.columns = ['date', 'sector', 'total_weight']
    
    merged_valid = merged_valid.merge(daily_sector_weight, on=['date', 'sector'])
    merged_valid['normalized_weight'] = merged_valid['weight'] / merged_valid['total_weight']
    
    # Step 5: 计算加权sentiment
    sector_signals = merged_valid.groupby(['date', 'sector']).agg({
        'sentiment': lambda x: (x * merged_valid.loc[x.index, 'normalized_weight']).sum(),
        'weight': 'sum',  # 原始权重和
        'qid': 'nunique',
        'document_id': 'nunique',
        'weight_lag_days': 'mean'
    }).reset_index()
    
    sector_signals.columns = [
        'date', 'sector', 'weighted_sentiment', 
        'raw_weight_sum', 'n_stocks', 'n_documents', 'avg_weight_lag'
    ]
    
    # Step 6: 计算覆盖率（相对于理论sector权重）
    # 获取每个sector的理论总权重（最近一次的完整权重）
    latest_sector_weights = sector_df.groupby('sector').apply(
        lambda x: x.nlargest(1, 'date')['weight'].sum()
    ).reset_index()
    latest_sector_weights.columns = ['sector', 'theoretical_weight']
    
    sector_signals = sector_signals.merge(latest_sector_weights, on='sector', how='left')
    sector_signals['coverage_ratio'] = sector_signals['raw_weight_sum'] / sector_signals['theoretical_weight']
    
    # Step 7: 详细诊断
    diagnostics = {
        'total_broker_records': len(doc_sentiment),
        'matched_records': len(merged_valid),
        'match_rate': len(merged_valid) / len(doc_sentiment),
        
        'unique_dates': len(sector_signals['date'].unique()),
        'date_range': (sector_signals['date'].min(), sector_signals['date'].max()),
        
        'avg_weight_lag_days': merged_valid['weight_lag_days'].mean(),
        'max_weight_lag_days': merged_valid['weight_lag_days'].max(),
        
        'coverage_by_sector': sector_signals.groupby('sector').agg({
            'coverage_ratio': 'mean',
            'n_stocks': 'mean',
            'weighted_sentiment': 'count'
        }).to_dict(),
        
        'weight_lag_distribution': merged_valid['weight_lag_days'].describe().to_dict()
    }
    
    return sector_signals, merged_valid, diagnostics


def analyze_signal_quality(sector_signals, min_coverage=0.3, min_stocks=5):
    """
    分析信号质量并过滤
    """
    
    # 质量标准
    sector_signals['is_high_quality'] = (
        (sector_signals['coverage_ratio'] >= min_coverage) &
        (sector_signals['n_stocks'] >= min_stocks)
    )
    
    print("\n=== 信号质量分析 ===")
    print(f"总信号数: {len(sector_signals)}")
    print(f"高质量信号数: {sector_signals['is_high_quality'].sum()}")
    print(f"高质量率: {sector_signals['is_high_quality'].mean():.2%}")
    
    print("\n按Sector统计:")
    quality_by_sector = sector_signals.groupby('sector').agg({
        'is_high_quality': ['sum', 'mean'],
        'coverage_ratio': 'mean',
        'n_stocks': 'mean',
        'avg_weight_lag': 'mean'
    }).round(3)
    print(quality_by_sector)
    
    # 时间序列分析
    print("\n按时间统计 (最近10个日期):")
    recent_quality = sector_signals.nlargest(10, 'date').groupby('date').agg({
        'sector': 'count',
        'is_high_quality': 'sum',
        'coverage_ratio': 'mean',
        'n_stocks': 'mean'
    }).round(3)
    print(recent_quality)
    
    return sector_signals[sector_signals['is_high_quality']]


def visualize_coverage(sector_signals, merged_valid):
    """
    可视化覆盖率和权重滞后
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 权重滞后分布
    axes[0, 0].hist(merged_valid['weight_lag_days'], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Weight Lag (Days)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Weight Lag Days')
    axes[0, 0].axvline(merged_valid['weight_lag_days'].median(), 
                       color='red', linestyle='--', label=f'Median: {merged_valid["weight_lag_days"].median():.0f}d')
    axes[0, 0].legend()
    
    # 2. 覆盖率分布
    axes[0, 1].hist(sector_signals['coverage_ratio'], bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Coverage Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Coverage Ratio by Sector-Date')
    axes[0, 1].axvline(0.3, color='red', linestyle='--', label='Threshold: 0.3')
    axes[0, 1].legend()
    
    # 3. 按sector的平均覆盖率
    coverage_by_sector = sector_signals.groupby('sector')['coverage_ratio'].mean().sort_values()
    coverage_by_sector.plot(kind='barh', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Average Coverage Ratio')
    axes[1, 0].set_title('Average Coverage by Sector')
    axes[1, 0].axvline(0.3, color='red', linestyle='--')
    
    # 4. 时间序列：每日有效sector数
    daily_sectors = sector_signals.groupby('date').agg({
        'sector': 'nunique',
        'coverage_ratio': 'mean'
    })
    
    ax4_1 = axes[1, 1]
    ax4_2 = ax4_1.twinx()
    
    ax4_1.plot(daily_sectors.index, daily_sectors['sector'], 'b-', label='# Sectors')
    ax4_2.plot(daily_sectors.index, daily_sectors['coverage_ratio'], 'r-', label='Avg Coverage')
    
    ax4_1.set_xlabel('Date')
    ax4_1.set_ylabel('Number of Sectors', color='b')
    ax4_2.set_ylabel('Average Coverage', color='r')
    ax4_1.set_title('Daily Sector Count and Coverage')
    
    plt.tight_layout()
    plt.savefig('/home/claude/sector_signal_diagnostics.png', dpi=300, bbox_inches='tight')
    print("\n诊断图表已保存到: /home/claude/sector_signal_diagnostics.png")
    
    return fig


# 完整执行流程
if __name__ == "__main__":
    # Step 1: 运行聚合
    sector_signals, merged_valid, diagnostics = aggregate_with_asof_merge(
        broker_df, 
        sector_df,
        max_lag_days=90  # 允许90天内的权重
    )
    
    # Step 2: 打印诊断信息
    print("\n=== 数据匹配诊断 ===")
    print(f"Broker记录数: {diagnostics['total_broker_records']:,}")
    print(f"成功匹配: {diagnostics['matched_records']:,} ({diagnostics['match_rate']:.1%})")
    print(f"\n生成信号的日期数: {diagnostics['unique_dates']:,}")
    print(f"日期范围: {diagnostics['date_range'][0]} 到 {diagnostics['date_range'][1]}")
    print(f"\n平均权重滞后: {diagnostics['avg_weight_lag_days']:.1f} 天")
    print(f"最大权重滞后: {diagnostics['max_weight_lag_days']:.0f} 天")
    
    # Step 3: 质量分析
    high_quality_signals = analyze_signal_quality(
        sector_signals,
        min_coverage=0.3,  # 至少30%的sector权重
        min_stocks=5        # 至少5只股票
    )
    
    # Step 4: 可视化
    fig = visualize_coverage(sector_signals, merged_valid)
    
    # Step 5: 导出结果
    high_quality_signals.to_csv('/home/claude/sector_signals_high_quality.csv', index=False)
    sector_signals.to_csv('/home/claude/sector_signals_all.csv', index=False)
    
    print(f"\n结果已保存:")
    print(f"- 所有信号: /home/claude/sector_signals_all.csv ({len(sector_signals)} 条)")
    print(f"- 高质量信号: /home/claude/sector_signals_high_quality.csv ({len(high_quality_signals)} 条)")
