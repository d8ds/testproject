"""
1. 计算个股权重
首先在每个E内部，把shares归一化成权重：
"""
holdings = holdings.with_columns(
    (pl.col("shares") / pl.col("shares").sum().over("factset_entity_id")).alias("weight")
)

"""
2. Join两个df
把holdings和sentiment df通过ID关联
"""
joined = sentiment_df.join(holdings, on="qid", how="inner")

"""
3. 计算加权sentiment
这里有几种选择：
方案A：直接用shares加权，每次filing独立计算
"""
joined = joined.with_columns(
    (pl.col("sentiment") * pl.col("weight")).alias("weighted_sentiment")
)

etf_signal = joined.group_by(["factset_entity_id", "date", "content_type"]).agg(
    pl.col("weighted_sentiment").sum().alias("etf_sentiment"),
    pl.col("weight").sum().alias("coverage")  # 实际覆盖的权重之和，可用于归一化
)

"""
注意这里coverage很重要——某个ETF里不是所有成分股都有filing，所以实际权重之和可能小于1，你需要决定是否rescale
"""
etf_signal = etf_signal.with_columns(
    (pl.col("etf_sentiment") / pl.col("coverage")).alias("etf_sentiment_rescaled")
)

"""
方案B：先在股票层面聚合多个content_type，再加权
如果同一个document有多个content_type，你可能想先在股票层面合并
"""
# 先按qid+date聚合（例如用numOfSentences加权平均）
stock_sentiment = (
    sentiment_df
    .with_columns(
        (pl.col("sentiment") * pl.col("numOfSentences")).alias("weighted_sent")
    )
    .group_by(["qid", "date"])
    .agg(
        (pl.col("weighted_sent").sum() / pl.col("numOfSentences").sum()).alias("stock_sentiment")
    )
)

# 再join holdings并做ETF级别加权
etf_signal = (
    stock_sentiment
    .join(holdings, on="qid", how="inner")
    .with_columns(
        (pl.col("stock_sentiment") * pl.col("weight")).alias("weighted_sentiment")
    )
    .group_by(["factset_entity_id", "date"])
    .agg(
        (pl.col("weighted_sentiment").sum() / pl.col("weight").sum()).alias("etf_sentiment")
    )
)

# comple
# Step 1: 计算ETF内各股票权重
holdings = holdings.with_columns(
    (pl.col("shares") / pl.col("shares").sum().over("factset_entity_id")).alias("weight")
)

# Step 2: 先在股票层面聚合同一document的多个content_type
# 用numOfSentences作为权重合并不同content的sentiment
stock_doc_sentiment = (
    sentiment_df
    .with_columns(
        (pl.col("sentiment") * pl.col("numOfSentences")).alias("weighted_sent")
    )
    .group_by(["qid", "document_id", "date"])
    .agg(
        (pl.col("weighted_sent").sum() / pl.col("numOfSentences").sum()).alias("stock_sentiment"),
        pl.col("numOfSentences").sum().alias("total_sentences"),
        pl.col("numOfWords").sum().alias("total_words"),
    )
)

# Step 3: Join holdings
joined = stock_doc_sentiment.join(holdings, on="qid", how="inner")

# Step 4: 聚合到ETF level
etf_signal = (
    joined
    .with_columns(
        (pl.col("stock_sentiment") * pl.col("weight")).alias("weighted_sentiment")
    )
    .group_by(["factset_entity_id", "date"])
    .agg(
        pl.col("weighted_sentiment").sum().alias("etf_sentiment_raw"),
        pl.col("weight").sum().alias("coverage"),
    )
    # rescale：除以实际覆盖权重，消除coverage不足带来的衰减
    .with_columns(
        (pl.col("etf_sentiment_raw") / pl.col("coverage")).alias("etf_sentiment")
    )
    # 可选：过滤低覆盖率的日期
    .filter(pl.col("coverage") > 0.1)
    .sort(["factset_entity_id", "date"])
)
#=======
