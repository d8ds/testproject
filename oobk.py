import polars as pl
from typing import Optional, Literal

"""
Sector-level Sentiment聚合工具
用于将QID-level的sentiment通过weight加权聚合到sector-level
"""


def aggregate_qid_to_sector_sentiment(
    broker_report: pl.DataFrame,
    sector_info: pl.DataFrame,
    date_col: str = 'date',
    qid_col: str = 'qid',
    sentiment_col: str = 'sentiment',
    document_col: Optional[str] = 'document_id',
    weight_col: str = 'weight',
    sector_col: str = 'sector',
    doc_agg_method: Literal['mean', 'median', 'sum'] = 'mean',
    fill_missing_weights: Optional[float] = None,
) -> pl.DataFrame:
    """
    将QID-level sentiment聚合到sector-level
    
    核心功能：
    1. 使用asof join向前查找最近的sector权重信息
    2. 在document级别和qid级别进行两次聚合
    3. 按sector和日期进行加权聚合
    
    参数:
        broker_report: 包含qid, date, document_id, sentiment的DataFrame
        sector_info: 包含qid, date, weight, sector的DataFrame
        date_col: 日期列名
        qid_col: QID列名
        sentiment_col: sentiment列名
        document_col: document_id列名，如果为None则跳过document级别聚合
        weight_col: 权重列名
        sector_col: sector列名
        doc_agg_method: document级别聚合方法 ('mean', 'median', 'sum')
        fill_missing_weights: 如果某个qid没有历史权重信息，用这个值填充（None则过滤掉）
        
    返回:
        DataFrame包含: date, sector, sector_sentiment, total_weight, n_qids
    """
    
    # 确保日期类型正确
    broker_report = broker_report.with_columns(
        pl.col(date_col).cast(pl.Date)
    )
    sector_info = sector_info.with_columns(
        pl.col(date_col).cast(pl.Date)
    )
    
    # Step 1: 聚合document级别（如果有document_col）
    if document_col is not None:
        if doc_agg_method == 'mean':
            agg_expr = pl.col(sentiment_col).mean()
        elif doc_agg_method == 'median':
            agg_expr = pl.col(sentiment_col).median()
        else:  # sum
            agg_expr = pl.col(sentiment_col).sum()
            
        qid_sentiment = (
            broker_report
            .group_by([qid_col, date_col, document_col])
            .agg(agg_expr.alias('_doc_sentiment'))
            .group_by([qid_col, date_col])
            .agg(pl.col('_doc_sentiment').mean().alias('_qid_sentiment'))
        )
    else:
        # 直接在qid-date级别聚合
        qid_sentiment = (
            broker_report
            .group_by([qid_col, date_col])
            .agg(pl.col(sentiment_col).mean().alias('_qid_sentiment'))
        )
    
    # Step 2: 排序准备asof join
    qid_sentiment = qid_sentiment.sort([qid_col, date_col])
    sector_info = sector_info.sort([qid_col, date_col])
    
    # Step 3: ASOF join - 关键步骤！
    # strategy="backward"表示向前查找 (<=)
    merged = qid_sentiment.join_asof(
        sector_info.select([qid_col, date_col, weight_col, sector_col]),
        on=date_col,
        by=qid_col,
        strategy='backward'
    )
    
    # Step 4: 处理missing weights
    if fill_missing_weights is not None:
        merged = merged.with_columns([
            pl.col(weight_col).fill_null(fill_missing_weights),
            pl.col(sector_col).fill_null('Unknown')
        ])
    else:
        merged = merged.filter(pl.col(sector_col).is_not_null())
    
    # Step 5: 加权聚合到sector level
    sector_sentiment = (
        merged
        .with_columns(
            (pl.col('_qid_sentiment') * pl.col(weight_col)).alias('_weighted_sentiment')
        )
        .group_by([date_col, sector_col])
        .agg([
            pl.col('_weighted_sentiment').sum().alias('_weighted_sum'),
            pl.col(weight_col).sum().alias('total_weight'),
            pl.col(qid_col).n_unique().alias('n_qids'),
            pl.col(qid_col).alias('qids_list')
        ])
        .with_columns(
            (pl.col('_weighted_sum') / pl.col('total_weight')).alias('sector_sentiment')
        )
        .select([date_col, sector_col, 'sector_sentiment', 'total_weight', 'n_qids', 'qids_list'])
        .sort([date_col, sector_col])
    )
    
    return sector_sentiment


def create_sector_signal_panel(
    sector_sentiment: pl.DataFrame,
    date_col: str = 'date',
    sector_col: str = 'sector',
    value_col: str = 'sector_sentiment',
    fill_strategy: Literal['forward', 'backward', 'zero', 'null'] = 'forward'
) -> pl.DataFrame:
    """
    将sector sentiment转换为完整的panel格式（所有日期×所有sector）
    
    参数:
        sector_sentiment: aggregate_qid_to_sector_sentiment的输出
        fill_strategy: 缺失值填充策略
            - 'forward': 向前填充（使用最近的历史值）
            - 'backward': 向后填充
            - 'zero': 填充0
            - 'null': 保持null
    """
    
    # 获取所有日期和所有sector
    all_dates = sector_sentiment.select(date_col).unique().sort(date_col)
    all_sectors = sector_sentiment.select(sector_col).unique()
    
    # 创建完整的日期×sector笛卡尔积
    panel = all_dates.join(all_sectors, how='cross')
    
    # Join回原始数据
    panel = panel.join(
        sector_sentiment.select([date_col, sector_col, value_col]),
        on=[date_col, sector_col],
        how='left'
    )
    
    # 填充策略
    if fill_strategy == 'forward':
        panel = panel.with_columns(
            pl.col(value_col).forward_fill().over(sector_col)
        )
    elif fill_strategy == 'backward':
        panel = panel.with_columns(
            pl.col(value_col).backward_fill().over(sector_col)
        )
    elif fill_strategy == 'zero':
        panel = panel.with_columns(
            pl.col(value_col).fill_null(0)
        )
    # 'null': do nothing
    
    return panel.sort([date_col, sector_col])


def add_signal_features(
    sector_sentiment: pl.DataFrame,
    date_col: str = 'date',
    sector_col: str = 'sector',
    value_col: str = 'sector_sentiment',
    ma_windows: Optional[list[int]] = None,
    diff_lags: Optional[list[int]] = None,
    zscore_window: Optional[int] = None
) -> pl.DataFrame:
    """
    为sector sentiment添加常用的信号特征
    
    参数:
        ma_windows: 移动平均窗口列表，例如 [5, 10, 20]
        diff_lags: 差分lag列表，例如 [1, 5] 表示1日变化和5日变化
        zscore_window: rolling z-score的窗口大小
    """
    
    result = sector_sentiment.sort([sector_col, date_col])
    
    # 移动平均
    if ma_windows:
        for window in ma_windows:
            result = result.with_columns(
                pl.col(value_col)
                .rolling_mean(window_size=window)
                .over(sector_col)
                .alias(f'{value_col}_ma{window}')
            )
    
    # 差分/变化
    if diff_lags:
        for lag in diff_lags:
            result = result.with_columns(
                (pl.col(value_col) - pl.col(value_col).shift(lag))
                .over(sector_col)
                .alias(f'{value_col}_diff{lag}')
            )
    
    # Rolling z-score
    if zscore_window:
        result = result.with_columns([
            pl.col(value_col).rolling_mean(window_size=zscore_window).over(sector_col).alias('_rolling_mean'),
            pl.col(value_col).rolling_std(window_size=zscore_window).over(sector_col).alias('_rolling_std')
        ]).with_columns(
            ((pl.col(value_col) - pl.col('_rolling_mean')) / pl.col('_rolling_std'))
            .alias(f'{value_col}_zscore{zscore_window}')
        ).drop(['_rolling_mean', '_rolling_std'])
    
    return result


# ============================================================================
# 实际使用示例
# ============================================================================

if __name__ == '__main__':
    # 创建示例数据
    broker_report = pl.DataFrame({
        'qid': ['Q001', 'Q001', 'Q001', 'Q002', 'Q002', 'Q003', 'Q001', 'Q002'],
        'date': [
            '2024-01-15', '2024-01-15', '2024-01-20', 
            '2024-01-18', '2024-01-25', '2024-01-22',
            '2024-01-28', '2024-02-01'
        ],
        'document_id': ['DOC1', 'DOC1', 'DOC2', 'DOC3', 'DOC4', 'DOC5', 'DOC6', 'DOC7'],
        'sentiment': [0.5, 0.6, 0.7, 0.3, 0.4, 0.8, 0.65, 0.5]
    })
    
    sector_info = pl.DataFrame({
        'qid': ['Q001', 'Q001', 'Q002', 'Q002', 'Q003', 'Q003'],
        'date': [
            '2024-01-10', '2024-01-25',
            '2024-01-15', '2024-01-30',
            '2024-01-20', '2024-02-01'
        ],
        'weight': [0.3, 0.4, 0.5, 0.6, 0.2, 0.25],
        'sector': ['Tech', 'Tech', 'Finance', 'Finance', 'Healthcare', 'Healthcare']
    })
    
    print("=" * 80)
    print("示例1: 基本聚合")
    print("=" * 80)
    
    result = aggregate_qid_to_sector_sentiment(broker_report, sector_info)
    print(result)
    
    print("\n" + "=" * 80)
    print("示例2: 添加信号特征")
    print("=" * 80)
    
    result_with_features = add_signal_features(
        result,
        ma_windows=[3, 5],
        diff_lags=[1],
        zscore_window=3
    )
    print(result_with_features)
    
    print("\n" + "=" * 80)
    print("示例3: 转换为完整的panel格式")
    print("=" * 80)
    
    panel = create_sector_signal_panel(result, fill_strategy='forward')
    print(panel)
    
    print("\n" + "=" * 80)
    print("示例4: Pivot为宽格式（时间序列）")
    print("=" * 80)
    
    wide_format = panel.pivot(
        values='sector_sentiment',
        index='date',
        on='sector'  # 使用on而不是columns
    ).sort('date')
    print(wide_format)
