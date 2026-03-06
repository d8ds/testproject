def convert_with_explicit_enddate(df_sector):
    """
    将endDate设为下一段的startDate - 1天
    适用于rebalancing日期不连续的情况
    """
    
    interval_df = (
        df_sector
        .sort(['sectorId', 'qid', 'date'])
        .with_columns([
            pl.col('weight')
              .shift(1)
              .over(['sectorId', 'qid'])
              .alias('prev_weight')
        ])
        .with_columns([
            (
                (pl.col('weight') != pl.col('prev_weight')) |
                pl.col('prev_weight').is_null()
            ).alias('is_change')
        ])
        .with_columns([
            pl.col('is_change')
              .cum_sum()
              .over(['sectorId', 'qid'])
              .alias('change_group')
        ])
        .group_by(['sectorId', 'qid', 'change_group', 'weight'])
        .agg([
            pl.col('date').min().alias('startDate')
        ])
        .sort(['sectorId', 'qid', 'startDate'])
        # 添加endDate：下一个startDate - 1天
        .with_columns([
            pl.col('startDate')
              .shift(-1)
              .over(['sectorId', 'qid'])
              .alias('next_startDate')
        ])
        .with_columns([
            pl.when(pl.col('next_startDate').is_not_null())
              .then(pl.col('next_startDate') - pl.duration(days=1))
              .otherwise(pl.date(2099, 12, 31))  # 最后一段，用一个很远的未来日期
              .alias('endDate')
        ])
        .select(['sectorId', 'qid', 'startDate', 'endDate', 'weight'])
    )
    
    return interval_df


def create_signal_with_rebalancing(
    df_company,
    df_sector,  # 包含startDate和endDate列
    lookback_days=30
):
    """
    处理公司在sector中的进出
    df_sector应该有：[sectorId, qid, startDate, endDate, weight]
    """
    
    # Step 1: 为每个sentiment，检查公司是否在sector内
    sentiment_valid = (
        df_company
        .join(
            df_sector,
            left_on='id',
            right_on='qid',
            how='left'
        )
        .filter(
            (pl.col('date') >= pl.col('startDate')) &
            (pl.col('date') <= pl.col('endDate'))
        )
    )
    
    # 后续同方法B
    # ...


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
