import pandas as pd
import numpy as np
from tqdm import tqdm

def aggregate_with_manual_lookup(broker_df, sector_df, max_lag_days=90):
    """
    手动实现向后查找逻辑，避免merge_asof的排序问题
    """
    
    # Step 1: 预处理
    broker_df = broker_df.copy()
    sector_df = sector_df.copy()
    
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    
    # 清理
    broker_df = broker_df.dropna(subset=['date', 'qid', 'sentiment'])
    sector_df = sector_df.dropna(subset=['date', 'qid', 'weight', 'sector'])
    
    print(f"Broker记录: {len(broker_df):,}")
    print(f"Sector记录: {len(sector_df):,}")
    
    # 聚合document级别
    doc_sentiment = broker_df.groupby(
        ['qid', 'date', 'document_id']
    )['sentiment'].mean().reset_index()
    
    print(f"聚合后记录: {len(doc_sentiment):,}\n")
    
    # Step 2: 为每个qid创建权重查找字典
    print("构建权重查找索引...")
    
    # 按qid分组，为每个qid创建日期->权重的映射
    sector_by_qid = {}
    for qid in tqdm(sector_df['qid'].unique(), desc="构建索引"):
        qid_data = sector_df[sector_df['qid'] == qid].sort_values('date')
        sector_by_qid[qid] = qid_data[['date', 'weight', 'sector']].values  # numpy array更快
    
    print(f"✓ 索引构建完成，覆盖{len(sector_by_qid):,}个qid\n")
    
    # Step 3: 手动查找每条记录的权重
    print("查找最近权重...")
    
    results = []
    max_lag_timedelta = pd.Timedelta(days=max_lag_days)
    
    for idx, row in tqdm(doc_sentiment.iterrows(), total=len(doc_sentiment), desc="匹配权重"):
        qid = row['qid']
        date = row['date']
        
        # 检查这个qid是否有权重信息
        if qid not in sector_by_qid:
            continue
        
        # 获取这个qid的所有权重记录
        qid_weights = sector_by_qid[qid]
        
        # 找到最近的、不晚于当前日期的权重
        # qid_weights: [date, weight, sector]
        valid_weights = qid_weights[qid_weights[:, 0] <= date]
        
        if len(valid_weights) == 0:
            continue
        
        # 取最近的一条
        latest_idx = -1  # 已经排序，最后一条就是最近的
        weight_date, weight, sector = valid_weights[latest_idx]
        
        # 检查时间差
        lag = date - weight_date
        if lag > max_lag_timedelta:
            continue
        
        results.append({
            'qid': qid,
            'date': date,
            'document_id': row['document_id'],
            'sentiment': row['sentiment'],
            'weight': weight,
            'sector': sector,
            'weight_date': weight_date,
            'weight_lag_days': lag.days
        })
    
    print(f"\n✓ 匹配完成：{len(results):,} / {len(doc_sentiment):,} ({len(results)/len(doc_sentiment):.1%})\n")
    
    if len(results) == 0:
        print("❌ 没有任何记录匹配到权重！")
        return None, None, None
    
    merged_valid = pd.DataFrame(results)
    
    # Step 4: 重新标准化权重
    print("重新标准化权重...")
    daily_sector_weight = merged_valid.groupby(['date', 'sector'])['weight'].sum().reset_index()
    daily_sector_weight.columns = ['date', 'sector', 'total_weight']
    
    merged_valid = merged_valid.merge(daily_sector_weight, on=['date', 'sector'])
    merged_valid['normalized_weight'] = merged_valid['weight'] / merged_valid['total_weight']
    
    # Step 5: 计算加权sentiment
    print("计算sector级别信号...\n")
    
    sector_signals = merged_valid.groupby(['date', 'sector']).agg({
        'sentiment': lambda x: (x * merged_valid.loc[x.index, 'normalized_weight']).sum(),
        'weight': 'sum',
        'qid': 'nunique',
        'document_id': 'nunique',
        'weight_lag_days': 'mean'
    }).reset_index()
    
    sector_signals.columns = [
        'date', 'sector', 'weighted_sentiment', 
        'raw_weight_sum', 'n_stocks', 'n_documents', 'avg_weight_lag'
    ]
    
    # Step 6: 计算覆盖率
    latest_sector_weights = sector_df.groupby('sector')['weight'].sum().reset_index()
    latest_sector_weights.columns = ['sector', 'theoretical_weight']
    
    sector_signals = sector_signals.merge(latest_sector_weights, on='sector', how='left')
    sector_signals['coverage_ratio'] = sector_signals['raw_weight_sum'] / sector_signals['theoretical_weight']
    
    # Step 7: 诊断
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
print("Sector信号聚合 (手动查找版本)")
print("="*60 + "\n")

sector_signals, merged_valid, diagnostics = aggregate_with_manual_lookup(
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
    print(f"\n生成信号日期数: {diagnostics['unique_dates']:,}")
    print(f"日期范围: {diagnostics['date_range'][0].date()} 到 {diagnostics['date_range'][1].date()}")
    print(f"\n权重滞后统计:")
    print(f"  中位数: {diagnostics['median_weight_lag_days']:.0f} 天")
    print(f"  平均值: {diagnostics['avg_weight_lag_days']:.1f} 天")
    print(f"  最大值: {diagnostics['max_weight_lag_days']:.0f} 天")
    print(f"\nSector覆盖:")
    print(f"  总数: {diagnostics['n_sectors']}")
    print(f"  平均覆盖率: {diagnostics['avg_coverage']:.1%}")
    print(f"  平均每信号股票数: {diagnostics['avg_stocks_per_signal']:.1f}")
    
    print("\n" + "="*60)
    print("信号示例 (前10条)")
    print("="*60)
    display_cols = ['date', 'sector', 'weighted_sentiment', 'n_stocks', 
                    'coverage_ratio', 'avg_weight_lag']
    print(sector_signals[display_cols].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("按Sector统计")
    print("="*60)
    sector_stats = sector_signals.groupby('sector').agg({
        'weighted_sentiment': 'count',
        'coverage_ratio': 'mean',
        'n_stocks': 'mean'
    }).round(3)
    sector_stats.columns = ['n_signals', 'avg_coverage', 'avg_stocks']
    print(sector_stats.sort_values('n_signals', ascending=False))
