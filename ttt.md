import polars as pl

# ── 1. 计算每个 doc_date 对应的 effective_report_date ──────────────────────
# join_asof backward: 找最近的 report_date <= doc_date
# 若 doc_date 早于所有 report_date，结果为 null → 填充为最早快照

min_report_date = etf_df["report_date"].min()

# 用唯一日期做映射，避免对整个大表做 asof join（性能优化）
unique_doc_dates = (
    sentiment_df
    .select("date")
    .unique()
    .sort("date")
)

unique_report_dates = (
    etf_df
    .select("report_date")
    .unique()
    .sort("report_date")
)

date_mapping = unique_doc_dates.join_asof(
    unique_report_dates.rename({"report_date": "effective_report_date"}),
    left_on="date",
    right_on="effective_report_date",
    strategy="backward",       # 找最近的历史快照
).with_columns(
    pl.col("effective_report_date").fill_null(min_report_date)  # 2017前填最早
)

# ── 2. sentiment_df 打上 effective_report_date ─────────────────────────────
sentiment_with_eff = sentiment_df.join(date_mapping, on="date")

# ── 3. join ETF holdings（qid + effective_report_date 双键）──────────────────
merged = sentiment_with_eff.join(
    etf_df.select(["factset_entity_id", "qid", "shares", "report_date"]),
    left_on=["qid", "effective_report_date"],
    right_on=["qid", "report_date"],
    how="inner",  # 只保留 ETF 中持有的个股
)

# ── 4. document 内部聚合：不同 content_type 按 numOfSentences 加权 ──────────
# 得到 (factset_entity_id, qid, document_id, date, shares) → doc_sentiment
doc_level = (
    merged
    .group_by(["factset_entity_id", "qid", "document_id", "date", "shares"])
    .agg(
        doc_sentiment=(
            (pl.col("sentiment") * pl.col("numOfSentences")).sum()
            / pl.col("numOfSentences").sum()
        ),
        total_sentences=pl.col("numOfSentences").sum(),
    )
)

# ── 5. ETF level 聚合：按 shares 加权各股 sentiment ────────────────────────
# 注意：同一 ETF 同一天可能有多只股票各自有文件，这里是跨 qid 聚合
etf_signal = (
    doc_level
    .group_by(["factset_entity_id", "document_id", "date"])
    .agg(
        etf_sentiment=(
            (pl.col("doc_sentiment") * pl.col("shares")).sum()
            / pl.col("shares").sum()
        ),
        covered_shares=pl.col("shares").sum(),  # 有文件覆盖的总持仓（可用于质量过滤）
    )
)
