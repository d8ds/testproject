import pandas as pd
import numpy as np

def aggregate_sentiment_to_sector(
    broker_df, 
    sector_df, 
    method='forward_fill',
    max_days=5,
    min_coverage=0.5
):
    """
    将qid级别的sentiment聚合到sector级别
    
    Parameters:
    -----------
    broker_df : DataFrame with columns [qid, date, document_id, sentiment]
    sector_df : DataFrame with columns [qid, date, weight, sector]
    method : str
        - 'forward_fill': 使用最近的权重并前向填充
        - 'renormalize': 重新标准化可用股票的权重
        - 'window': 使用时间窗口匹配
    max_days : int, 时间窗口大小（仅用于window方法）
    min_coverage : float, 最小覆盖率阈值（0-1）
    
    Returns:
    --------
    sector_signals : DataFrame with [date, sector, weighted_sentiment, coverage, n_stocks]
    diagnostics : dict with coverage statistics
    """
    
    # Step 1: 先处理document级别的聚合（如果一个doc_id有多条记录）
    # 选择聚合方式：mean, median, last等
    doc_sentiment = broker_df.groupby(
        ['qid', 'date', 'document_id']
    )['sentiment'].mean().reset_index()
    
    # Step 2: 准备sector权重数据
    sector_df = sector_df.copy()
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    doc_sentiment['date'] = pd.to_datetime(doc_sentiment['date'])
    
    if method == 'forward_fill':
        # 方法1: Forward fill weights
        # 为每个qid创建完整的日期序列
        sector_pivot = sector_df.pivot_table(
            index='date', 
            columns='qid', 
            values=['weight', 'sector'],
            aggfunc='first'
        )
        
        # Forward fill weights
        sector_pivot['weight'] = sector_pivot['weight'].ffill(limit=max_days)
        sector_pivot['sector'] = sector_pivot['sector'].ffill(limit=max_days)
        
        # 转回long format
        sector_filled = sector_pivot.stack(level=1).reset_index()
        sector_filled.columns = ['date', 'qid', 'weight', 'sector']
        sector_filled = sector_filled.dropna()
        
        # 合并sentiment和sector信息
        merged = doc_sentiment.merge(
            sector_filled,
            on=['qid', 'date'],
            how='inner'
        )
        
    elif method == 'renormalize':
        # 方法2: 只使用可用的股票，重新标准化权重
        merged = doc_sentiment.merge(
            sector_df,
            on=['qid', 'date'],
            how='inner'
        )
        
        # 计算每个sector-date组合的实际权重总和
        weight_sums = merged.groupby(['date', 'sector'])['weight'].sum().reset_index()
        weight_sums.columns = ['date', 'sector', 'available_weight_sum']
        
        merged = merged.merge(weight_sums, on=['date', 'sector'])
        
        # 重新标准化权重
        merged['normalized_weight'] = merged['weight'] / merged['available_weight_sum']
        merged['weight'] = merged['normalized_weight']
        
    elif method == 'window':
        # 方法3: 时间窗口匹配
        merged_list = []
        
        for idx, row in doc_sentiment.iterrows():
            qid, date, doc_id, sentiment = row['qid'], row['date'], row['document_id'], row['sentiment']
            
            # 在时间窗口内查找权重
            window_start = date - pd.Timedelta(days=max_days)
            window_end = date + pd.Timedelta(days=max_days)
            
            weight_match = sector_df[
                (sector_df['qid'] == qid) & 
                (sector_df['date'] >= window_start) & 
                (sector_df['date'] <= window_end)
            ].sort_values('date', ascending=False).head(1)
            
            if not weight_match.empty:
                merged_list.append({
                    'qid': qid,
                    'date': date,
                    'document_id': doc_id,
                    'sentiment': sentiment,
                    'weight': weight_match.iloc[0]['weight'],
                    'sector': weight_match.iloc[0]['sector'],
                    'weight_date': weight_match.iloc[0]['date']
                })
        
        merged = pd.DataFrame(merged_list)
    
    # Step 3: 计算sector级别的加权sentiment
    sector_signals = merged.groupby(['date', 'sector']).apply(
        lambda x: pd.Series({
            'weighted_sentiment': (x['sentiment'] * x['weight']).sum() / x['weight'].sum(),
            'coverage': x['weight'].sum(),  # 实际覆盖的权重
            'n_stocks': len(x['qid'].unique()),
            'n_documents': len(x['document_id'].unique())
        })
    ).reset_index()
    
    # Step 4: 过滤低覆盖率的信号
    sector_signals['pass_coverage'] = sector_signals['coverage'] >= min_coverage
    
    # Step 5: 生成诊断信息
    diagnostics = {
        'total_dates': len(sector_signals['date'].unique()),
        'avg_coverage_by_sector': sector_signals.groupby('sector')['coverage'].mean().to_dict(),
        'low_coverage_count': (~sector_signals['pass_coverage']).sum(),
        'missing_rate_by_sector': sector_signals.groupby('sector')['pass_coverage'].apply(
            lambda x: 1 - x.mean()
        ).to_dict()
    }
    
    return sector_signals, diagnostics


# 使用示例和诊断
def diagnose_coverage(broker_df, sector_df):
    """诊断数据覆盖情况"""
    
    # 转换日期
    broker_df['date'] = pd.to_datetime(broker_df['date'])
    sector_df['date'] = pd.to_datetime(sector_df['date'])
    
    # 按日期统计
    broker_dates = set(broker_df['date'].unique())
    sector_dates = set(sector_df['date'].unique())
    
    print(f"Broker report日期数: {len(broker_dates)}")
    print(f"Sector信息日期数: {len(sector_dates)}")
    print(f"重叠日期数: {len(broker_dates & sector_dates)}")
    print(f"仅broker有的日期数: {len(broker_dates - sector_dates)}")
    
    # 按qid统计
    broker_qids = set(broker_df['qid'].unique())
    sector_qids = set(sector_df['qid'].unique())
    
    print(f"\nBroker report覆盖的qid数: {len(broker_qids)}")
    print(f"Sector信息覆盖的qid数: {len(sector_qids)}")
    print(f"重叠qid数: {len(broker_qids & sector_qids)}")
    
    # 每日匹配率
    daily_match = []
    for date in broker_dates:
        broker_qids_day = set(broker_df[broker_df['date'] == date]['qid'])
        sector_qids_day = set(sector_df[sector_df['date'] == date]['qid'])
        
        if len(broker_qids_day) > 0:
            match_rate = len(broker_qids_day & sector_qids_day) / len(broker_qids_day)
            daily_match.append({
                'date': date,
                'match_rate': match_rate,
                'broker_qids': len(broker_qids_day),
                'matched_qids': len(broker_qids_day & sector_qids_day)
            })
    
    match_df = pd.DataFrame(daily_match)
    print(f"\n平均每日匹配率: {match_df['match_rate'].mean():.2%}")
    print(f"最低每日匹配率: {match_df['match_rate'].min():.2%}")
    
    return match_df


# 使用示例
if __name__ == "__main__":
    # 先诊断
    coverage_df = diagnose_coverage(broker_df, sector_df)
    
    # 选择合适的方法进行聚合
    sector_signals, diagnostics = aggregate_sentiment_to_sector(
        broker_df, 
        sector_df,
        method='forward_fill',  # 或 'renormalize', 'window'
        max_days=5,
        min_coverage=0.5
    )
    
    print("\n诊断结果:")
    print(diagnostics)
    
    # 查看结果
    print("\nSector信号示例:")
    print(sector_signals[sector_signals['pass_coverage']].head(10))
