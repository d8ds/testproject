import pandas as pd
import numpy as np

def aggregate_with_asof_merge(broker_df, sector_df, max_lag_days=90):
    """
    使用as-of merge策略：为每个broker report找到最近的sector权重
    """
    
    # Step 1: 深度复制并清理
    broker_df = broker_df.copy()
    sector_df = sector_df.copy()
    
    # 统一datetime格式
    broker_df['date'] = pd.to_datetime(broker_df['date']).astype('datetime64[ns]')
    sector_df['date'] = pd.to_datetime(sector_df['date']).astype('datetime64[ns]')
    
    # 🔧 关键：移除任何NaN日期或qid
    print("清理数据...")
    broker_df = broker_df.dropna(subset=['date', 'qid'])
    sector_df = sector_df.dropna(subset=['date', 'qid'])
    
    print(f"Broker记录: {len(broker_df):,}")
    print(f"Sector记录: {len(sector_df):,}")
    
    # 聚合document级别
    doc_sentiment = broker_df.groupby(
        ['qid', 'date', 'document_id']
    )['sentiment'].mean().reset_index()
    
    print(f"\n聚合后的Broker记录: {len(doc_sentiment):,}")
    
    # 🔧 关键修复：确保qid是字符串类型（避免类型不匹配）
    doc_sentiment['qid'] = doc_sentiment['qid'].astype(str)
    sector_df['qid'] = sector_df['qid'].astype(str)
    
    # 🔧 严格排序：先按qid，再按date
    print("\n严格排序数据...")
    doc_sentiment = doc_sentiment.sort_values(
        ['qid', 'date'], 
        ascending=[True, True]
    ).reset_index(drop=True)
    
    sector_df = sector_df.sort_values(
        ['qid', 'date'], 
        ascending=[True, True]
    ).reset_index(drop=True)
    
    # 🔧 验证排序（更严格）
    print("验证排序...")
    
    # 检查是否有重复的qid导致排序问题
    doc_qid_check = doc_sentiment.groupby('qid')['date'].apply(
        lambda x: x.is_monotonic_increasing
    )
    sector_qid_check = sector_df.groupby('qid')['date'].apply(
        lambda x: x.is_monotonic_increasing
    )
    
    if not doc_qid_check.all():
        bad_qids = doc_qid_check[~doc_qid_check].index.tolist()
        print(f"⚠️  警告：doc_sentiment中有{len(bad_qids)}个qid的日期未正确排序")
        print(f"示例qid: {bad_qids[:5]}")
        # 强制重排
        doc_sentiment = doc_sentiment.sort_values(['qid', 'date']).reset_index(drop=True)
    
    if not sector_qid_check.all():
        bad_qids = sector_qid_check[~sector_qid_check].index.tolist()
        print(f"⚠️  警告：sector_df中有{len(bad_qids)}个qid的日期未正确排序")
        print(f"示例qid: {bad_qids[:5]}")
        # 强制重排
        sector_df = sector_df.sort_values(['qid', 'date']).reset_index(drop=True)
    
    print("✓ 排序验证通过\n")
    
    # 🔧 再次确认排序（pandas有时会有bug）
    # 使用sort_index确保完全排序
    doc_sentiment = doc_sentiment.sort_values(['qid', 'date']).reset_index(drop=True)
    sector_df_merge = sector_df[['qid', 'date', 'weight', 'sector']].sort_values(
        ['qid', 'date']
    ).reset_index(drop=True)
    
    # Step 2: 执行merge_asof
    print("执行merge_asof...")
    try:
        merged = pd.merge_asof(
            doc_sentiment,
            sector_df_merge,
            on='date',
            by='qid',
            direction='backward',
            tolerance=pd.Timedelta(days=max_lag_days)
        )
        print("✓ Merge成功")
    except ValueError as e:
        print(f"❌ Merge失败: {e}")
        print("\n调试信息:")
        print(f"doc_sentiment shape: {doc_sentiment.shape}")
        print(f"sector_df_merge shape: {sector_df_merge.shape}")
        print(f"\ndoc_sentiment前5行:")
        print(doc_sentiment.head())
        print(f"\nsector_df_merge前5行:")
        print(sector_df_merge.head())
        
        # 尝试找出问题qid
        print("\n检查前10个qid的排序...")
        for qid in doc_sentiment['qid'].unique()[:10]:
            doc_dates = doc_sentiment[doc_sentiment['qid'] == qid]['date']
            sector_dates = sector_df_merge[sector_df_merge['qid'] == qid]['date']
            
            if not doc_dates.is_monotonic_increasing:
                print(f"  {qid}: doc dates NOT sorted")
            if len(sector_dates) > 0 and not sector_dates.is_monotonic_increasing:
                print(f"  {qid}: sector dates NOT sorted")
        
        return None, None, None
    
    print(f"匹配到权重的记录数: {merged['weight'].notna().sum():,} / {len(merged):,}")
    
    # Step 3: 简化的权重日期计算
    print("计算权重滞后天数...")
    
    # 为每个匹配的记录找到对应的权重日期
    # 方法：按qid-weight-sector组合合并
    sector_dates = sector_df[['qid', 'weight', 'sector', 'date']].drop_duplicates()
    sector_dates = sector_dates.rename(columns={'date': 'weight_date'})
    
    merged = merged.merge(
        sector_dates,
        on=['qid', 'weight', 'sector'],
        how='left'
    )
    
    # 处理可能的多个匹配（取最接近的日期）
    merged['date_diff'] = (merged['date'] - merged['weight_date']).abs()
    merged = merged.sort_values('date_diff').groupby(
        ['qid', 'date', 'document_id'], as_index=False
    ).first()
    
    merged['weight_lag_days'] = (merged['date'] - merged['weight_date']).dt.days
    merged = merged.drop(['date_diff'], axis=1, errors='ignore')
    
    # Step 4: 过滤有效记录
    merged_valid = merged[merged['weight'].notna()].copy()
    
    if len(merged_valid) == 0:
        print("❌ 没有任何记录匹配到权重！")
        return None, None, None
    
    print(f"✓ 有效记录数: {len(merged_valid):,}\n")
    
    # Step 5: 重新标准化权重
    print("重新标准化权重...")
    daily_sector_weight = merged_valid.groupby(['date', 'sector'])['weight'].sum().reset_index()
    daily_sector_weight.columns = ['date', 'sector', 'total_weight']
    
    merged_valid = merged_valid.merge(daily_sector_weight, on=['date', 'sector'])
    merged_valid['normalized_weight'] = merged_valid['weight'] / merged_valid['total_weight']
    
    # Step 6: 计算加权sentiment
    print("计算sector级别加权sentiment...\n")
    
    sector_signals = merged_valid.groupby(['date', 'sector']).apply(
        lambda x: pd.Series({
            'weighted_sentiment': (x['sentiment'] * x['normalized_weight']).sum(),
            'raw_weight_sum': x['weight'].sum(),
            'n_stocks': x['qid'].nunique(),
            'n_documents': x['document_id'].nunique(),
            'avg_weight_lag': x['weight_lag_days'].mean()
        })
    ).reset_index()
    
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


# 执行
print("="*60)
print("Sector信号聚合")
print("="*60)

sector_signals, merged_valid, diagnostics = aggregate_with_asof_merge(
    broker_df, 
    sector_df,
    max_lag_days=90
)

if sector_signals is not None:
    print("\n" + "="*60)
    print("诊断结果")
    print("="*60)
    print(f"总Broker记录: {diagnostics['total_broker_records']:,}")
    print(f"成功匹配: {diagnostics['matched_records']:,} ({diagnostics['match_rate']:.1%})")
    print(f"\n生成信号日期数: {diagnostics['unique_dates']:,}")
    print(f"日期范围: {diagnostics['date_range'][0].date()} 到 {diagnostics['date_range'][1].date()}")
    print(f"\n权重滞后: 中位数={diagnostics['median_weight_lag_days']:.0f}天, "
          f"平均={diagnostics['avg_weight_lag_days']:.1f}天")
    print(f"\nSector数: {diagnostics['n_sectors']}")
    print(f"平均覆盖率: {diagnostics['avg_coverage']:.1%}")
    
    print("\n示例:")
    print(sector_signals.head(10).to_string(index=False))
