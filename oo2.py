import pandas as pd
import numpy as np

def aggregate_with_asof_merge(broker_df, sector_df, max_lag_days=90):
    """
    使用as-of merge策略：为每个broker report找到最近的sector权重
    """
    
    # Step 1: 预处理
    broker_df = broker_df.copy()
    sector_df = sector_df.copy()
    
    # 统一datetime格式
    broker_df['date'] = pd.to_datetime(broker_df['date']).astype('datetime64[ns]')
    sector_df['date'] = pd.to_datetime(sector_df['date']).astype('datetime64[ns]')
    
    print(f"Broker date dtype: {broker_df['date'].dtype}")
    print(f"Sector date dtype: {sector_df['date'].dtype}")
    
    # 聚合document级别
    doc_sentiment = broker_df.groupby(
        ['qid', 'date', 'document_id']
    )['sentiment'].mean().reset_index()
    
    print(f"\n处理 {len(doc_sentiment):,} 条broker记录...")
    print(f"匹配 {len(sector_df):,} 条sector权重记录...\n")
    
    # 🔧 关键修复：确保正确排序
    doc_sentiment = doc_sentiment.sort_values(['qid', 'date']).reset_index(drop=True)
    sector_df = sector_df.sort_values(['qid', 'date']).reset_index(drop=True)
    
    # 验证排序
    print("验证排序...")
    assert doc_sentiment.groupby('qid')['date'].apply(lambda x: x.is_monotonic_increasing).all(), \
        "doc_sentiment未正确按qid-date排序"
    assert sector_df.groupby('qid')['date'].apply(lambda x: x.is_monotonic_increasing).all(), \
        "sector_df未正确按qid-date排序"
    print("排序验证通过 ✓\n")
    
    # Step 2: 使用merge_asof找最近的权重
    print("执行merge_asof...")
    merged = pd.merge_asof(
        doc_sentiment,
        sector_df[['qid', 'date', 'weight', 'sector']],
        on='date',
        by='qid',
        direction='backward',
        tolerance=pd.Timedelta(days=max_lag_days)
    )
    
    print(f"Merge完成，匹配到权重的记录数: {merged['weight'].notna().sum():,} / {len(merged):,}")
    
    # Step 3: 计算权重滞后天数（简化版本）
    # 创建sector_df的日期查找字典
    print("计算权重滞后天数...")
    
    # 为了效率，先创建一个辅助DataFrame记录每个sector记录的日期
    sector_lookup = sector_df[['qid', 'date', 'weight', 'sector']].copy()
    sector_lookup = sector_lookup.rename(columns={'date': 'weight_date'})
    
    # 通过weight和sector来反向查找权重日期
    merged = merged.merge(
        sector_lookup,
        on=['qid', 'weight', 'sector'],
        how='left'
    )
    
    # 如果有重复（一个qid-weight-sector组合对应多个日期），取最近的
    merged = merged.sort_values(['qid', 'date', 'weight_date'])
    merged['weight_date'] = merged.groupby(['qid', 'date', 'document_id'])['weight_date'].transform('last')
    merged = merged.drop_duplicates(subset=['qid', 'date', 'document_id'])
    
    merged['weight_lag_days'] = (merged['date'] - merged['weight_date']).dt.days
    
    # Step 4: 过滤掉没有匹配到权重的记录
    merged_valid = merged[merged['weight'].notna()].copy()
    
    if len(merged_valid) == 0:
        print("❌ 警告：没有任何记录匹配到权重！")
        return None, None, None
    
    print(f"✓ 有效记录数: {len(merged_valid):,}\n")
    
    # Step 5: 按date-sector重新标准化权重
    print("重新标准化权重...")
    daily_sector_weight = merged_valid.groupby(['date', 'sector'])['weight'].sum().reset_index()
    daily_sector_weight.columns = ['date', 'sector', 'total_weight']
    
    merged_valid = merged_valid.merge(daily_sector_weight, on=['date', 'sector'])
    merged_valid['normalized_weight'] = merged_valid['weight'] / merged_valid['total_weight']
    
    # Step 6: 计算加权sentiment
    print("计算sector级别加权sentiment...\n")
    
    # 使用更简单的方式计算加权平均
    def weighted_avg(group):
        return (group['sentiment'] * group['normalized_weight']).sum()
    
    sector_signals = merged_valid.groupby(['date', 'sector']).agg({
        'sentiment': weighted_avg,
        'weight': 'sum',
        'qid': 'nunique',
        'document_id': 'nunique',
        'weight_lag_days': 'mean'
    }).reset_index()
    
    sector_signals.columns = [
        'date', 'sector', 'weighted_sentiment', 
        'raw_weight_sum', 'n_stocks', 'n_documents', 'avg_weight_lag'
    ]
    
    # Step 7: 计算覆盖率
    latest_sector_weights = sector_df.groupby('sector')['weight'].sum().reset_index()
    latest_sector_weights.columns = ['sector', 'theoretical_weight']
    
    sector_signals = sector_signals.merge(latest_sector_weights, on='sector', how='left')
    sector_signals['coverage_ratio'] = sector_signals['raw_weight_sum'] / sector_signals['theoretical_weight']
    
    # Step 8: 诊断
    diagnostics = {
        'total_broker_records': len(doc_sentiment),
        'matched_records': len(merged_valid),
        'match_rate': len(merged_valid) / len(doc_sentiment),
        
        'unique_dates': len(sector_signals['date'].unique()),
        'date_range': (sector_signals['date'].min(), sector_signals['date'].max()),
        
        'avg_weight_lag_days': merged_valid['weight_lag_days'].mean(),
        'median_weight_lag_days': merged_valid['weight_lag_days'].median(),
        'max_weight_lag_days': merged_valid['weight_lag_days'].max(),
        
        'sectors': sorted(sector_signals['sector'].unique().tolist()),
        'n_sectors': sector_signals['sector'].nunique(),
        
        'avg_coverage': sector_signals['coverage_ratio'].mean(),
        'avg_stocks_per_signal': sector_signals['n_stocks'].mean()
    }
    
    return sector_signals, merged_valid, diagnostics


def analyze_signal_quality(sector_signals, min_coverage=0.3, min_stocks=5):
    """分析信号质量"""
    
    if sector_signals is None:
        return None
    
    # 质量标准
    sector_signals['is_high_quality'] = (
        (sector_signals['coverage_ratio'] >= min_coverage) &
        (sector_signals['n_stocks'] >= min_stocks)
    )
    
    print("\n" + "="*60)
    print("信号质量分析")
    print("="*60)
    print(f"总信号数: {len(sector_signals):,}")
    print(f"高质量信号数: {sector_signals['is_high_quality'].sum():,}")
    print(f"高质量率: {sector_signals['is_high_quality'].mean():.1%}")
    
    print("\n按Sector统计:")
    quality_by_sector = sector_signals.groupby('sector').agg({
        'weighted_sentiment': 'count',
        'is_high_quality': ['sum', 'mean'],
        'coverage_ratio': 'mean',
        'n_stocks': 'mean',
        'avg_weight_lag': 'mean'
    }).round(3)
    quality_by_sector.columns = ['n_signals', 'n_high_quality', 'pct_high_quality', 
                                  'avg_coverage', 'avg_stocks', 'avg_lag_days']
    print(quality_by_sector.sort_values('n_signals', ascending=False))
    
    return sector_signals[sector_signals['is_high_quality']]


# 完整执行
print("="*60)
print("开始Sector信号聚合")
print("="*60)

sector_signals, merged_valid, diagnostics = aggregate_with_asof_merge(
    broker_df, 
    sector_df,
    max_lag_days=90
)

if sector_signals is not None:
    print("="*60)
    print("诊断结果")
    print("="*60)
    print(f"总Broker记录: {diagnostics['total_broker_records']:,}")
    print(f"成功匹配: {diagnostics['matched_records']:,} ({diagnostics['match_rate']:.1%})")
    print(f"\n生成的唯一日期数: {diagnostics['unique_dates']:,}")
    print(f"日期范围: {diagnostics['date_range'][0].strftime('%Y-%m-%d')} 到 {diagnostics['date_range'][1].strftime('%Y-%m-%d')}")
    print(f"\n权重滞后统计:")
    print(f"  - 中位数: {diagnostics['median_weight_lag_days']:.0f} 天")
    print(f"  - 平均值: {diagnostics['avg_weight_lag_days']:.1f} 天")
    print(f"  - 最大值: {diagnostics['max_weight_lag_days']:.0f} 天")
    print(f"\nSector数量: {diagnostics['n_sectors']}")
    print(f"平均覆盖率: {diagnostics['avg_coverage']:.1%}")
    print(f"平均每信号股票数: {diagnostics['avg_stocks_per_signal']:.1f}")
    
    # 质量分析
    high_quality_signals = analyze_signal_quality(
        sector_signals,
        min_coverage=0.3,
        min_stocks=5
    )
    
    # 查看示例
    print("\n" + "="*60)
    print("信号示例 (最近5天)")
    print("="*60)
    recent = sector_signals.nlargest(50, 'date').head(10)
    print(recent[['date', 'sector', 'weighted_sentiment', 'n_stocks', 'coverage_ratio', 'avg_weight_lag']].to_string(index=False))
    
    print("\n✓ 处理完成！")
