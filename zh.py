def create_pit_sector_signal_explicit(
    df_company,
    df_sector,
    lookback_days=30
):
    """
    显式处理每个date的rolling window
    更慢但更清晰和可控
    """
    
    # Step 1: 扩展sector weight历史，forward fill到每一天
    sector_daily = (
        df_sector
        .sort('date')
        .group_by(['sectorId', 'qid'])
        .agg([
            pl.col('date'),
            pl.col('weight')
        ])
        .explode(['date', 'weight'])
        # Forward fill weight到下一次更新
        .with_columns([
            pl.col('weight').forward_fill().over(['sectorId', 'qid'])
        ])
    )
    
    # Step 2: Join sentiment with daily weights
    sentiment_daily = (
        df_company
        .join(
            sector_daily,
            left_on=['id', 'date'],
            right_on=['qid', 'date'],
            how='inner'  # 只保留在sector内的公司
        )
        .with_columns([
            (pl.col('sentiment') * pl.col('weight')).alias('weighted_sentiment')
        ])
    )
    
    # Step 3: 对每个(sectorId, date)，聚合过去N天的数据
    # 注意：这里的weight已经是point-in-time的了
    sector_signal = (
        sentiment_daily
        .sort('date')
        .group_by_dynamic(
            'date',
            every='1d',
            period=f'{lookback_days}d',
            by='sectorId'
        )
        .agg([
            # 在window内，每天的weight已经是正确的了
            pl.col('weighted_sentiment').sum(),
            pl.col('weight').sum(),
            pl.col('id').n_unique().alias('n_companies'),
        ])
        .with_columns([
            (pl.col('weighted_sentiment') / pl.col('weight')).alias('sector_sentiment')
        ])
    )
    
    return sector_signal
