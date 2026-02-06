import polars as pl
import numpy as np

# 假设你的df格式如下
# df: columns = [ID, date, value]

# 1. 计算1年移动平均
df_with_ma = df.sort(['ID', 'date']).with_columns([
    pl.col('value')
      .rolling_mean(
          window_size='365d',
          by='date',
          closed='left'  # 避免look-ahead bias
      )
      .over('ID')
      .alias('value_ma_1y')
])

# 2. 计算长期分位数（基于移动平均）
# 方法A: 全样本分位数
df_with_tiles = df_with_ma.with_columns([
    pl.col('value_ma_1y')
      .rank(method='average')
      .over('date')  # 每个日期横截面排序
      .alias('rank')
]).with_columns([
    (pl.col('rank') / pl.col('rank').max().over('date'))
      .alias('percentile')
])

# 方法B: 滚动历史分位数（更稳健）
df_with_tiles = df_with_ma.sort(['ID', 'date']).with_columns([
    pl.col('value_ma_1y')
      .rolling_quantile(
          quantile=0.5,  # 可以计算多个分位点
          window_size='730d',  # 2年历史窗口
          by='date'
      )
      .over('ID')
      .alias('median_2y')
])

# 3. 创建分位数信号（例如五分位）
df_final = df_with_tiles.with_columns([
    pl.when(pl.col('percentile') <= 0.2).then(1)
      .when(pl.col('percentile') <= 0.4).then(2)
      .when(pl.col('percentile') <= 0.6).then(3)
      .when(pl.col('percentile') <= 0.8).then(4)
      .otherwise(5)
      .alias('quintile')
])

# 4. 标准化为z-score形式（如果需要）
df_final = df_final.with_columns([
    ((pl.col('value_ma_1y') - pl.col('value_ma_1y').mean().over('date')) 
     / pl.col('value_ma_1y').std().over('date'))
     .alias('value_ma_zscore')
])
----
几点建议基于你的filing signal经验：

窗口选择: 你在8-K项目中发现90天窗口效果好，1年MA可能需要测试不同窗口（180d, 252d, 365d）
避免前视偏差: 使用closed='left'确保只用历史数据
处理缺失值:

# 确保足够的历史数据
df_filtered = df_with_ma.filter(
    pl.col('value_ma_1y').is_not_null()
)

=========

信号正交化: 如果你想将这个长期信号与短期信号组合：
# 计算residual signal
df_orthog = df_final.with_columns([
    (pl.col('value') - pl.col('value_ma_1y')).alias('short_term_residual')
])

