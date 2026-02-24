# 用 left join 看看有多少 null
merged_check = sentiment_with_eff.join(
    etf_df.select(["factset_entity_id", "qid", "shares", "report_date"]),
    left_on=["qid", "effective_report_date"],
    right_on=["qid", "report_date"],
    how="left",
)
print(f"Total rows: {merged_check.shape[0]}")
print(f"Matched rows: {merged_check.filter(pl.col('factset_entity_id').is_not_null()).shape[0]}")
print(f"factset_entity_id nunique after left join: {merged_check['factset_entity_id'].n_unique()}")

# 找一个 overlap qid，手动检查它在两个表里的日期
sample_qid = list(overlap_qids)[0]
print(sentiment_with_eff.filter(pl.col("qid") == sample_qid).select(["qid", "date", "effective_report_date"]).head())
print(etf_df.filter(pl.col("qid") == sample_qid).select(["qid", "report_date"]))


# 最关键的：effective_report_date 的值是否出现在 report_date 里
eff_dates = set(result1["effective_report_date"].to_list())
rep_dates = set(result2["report_date"].to_list())
print(f"effective_report_dates: {eff_dates}")
print(f"report_dates in etf: {rep_dates}")
print(f"overlap: {eff_dates & rep_dates}")


# backward：找最近的历史快照（2017年后的数据）
date_mapping_backward = unique_doc_dates.join_asof(
    unique_report_dates.rename({"report_date": "effective_report_date"}),
    left_on="date",
    right_on="effective_report_date",
    strategy="backward",
)

# forward：对 backward 结果为 null 的，找最近的未来快照
date_mapping_forward = unique_doc_dates.join_asof(
    unique_report_dates.rename({"report_date": "effective_report_date"}),
    left_on="date",
    right_on="effective_report_date",
    strategy="forward",
)

# 合并：backward 有结果就用 backward，否则用 forward
date_mapping = date_mapping_backward.with_columns(
    pl.when(pl.col("effective_report_date").is_null())
    .then(date_mapping_forward["effective_report_date"])
    .otherwise(pl.col("effective_report_date"))
    .alias("effective_report_date")
)

# 验证：不应该再有 null 了
print(date_mapping["effective_report_date"].null_count())
print(date_mapping["effective_report_date"].value_counts().sort("count", descending=True).head(10))

3======
import polars as pl

# ── 0. 统一日期类型 ────────────────────────────────────────────────────────
sentiment_df = sentiment_df.with_columns(pl.col("date").cast(pl.Date))
etf_df = etf_df.with_columns(pl.col("report_date").cast(pl.Date))

# ── 1. 构建 date → effective_report_date 映射 ──────────────────────────────
unique_doc_dates = sentiment_df.select("date").unique().sort("date")
unique_report_dates = (
    etf_df.select("report_date")
    .unique()
    .sort("report_date")
    .rename({"report_date": "effective_report_date"})
)

# backward：找最近的历史快照
date_mapping_backward = unique_doc_dates.join_asof(
    unique_report_dates,
    left_on="date",
    right_on="effective_report_date",
    strategy="backward",
)

# forward：对 2017 年之前 backward 为 null 的，找最近的未来快照
date_mapping_forward = unique_doc_dates.join_asof(
    unique_report_dates,
    left_on="date",
    right_on="effective_report_date",
    strategy="forward",
)

# 合并：优先用 backward，null 时用 forward
date_mapping = date_mapping_backward.with_columns(
    pl.when(pl.col("effective_report_date").is_null())
    .then(date_mapping_forward["effective_report_date"])
    .otherwise(pl.col("effective_report_date"))
    .alias("effective_report_date")
)

# 验证
assert date_mapping["effective_report_date"].null_count() == 0, "仍有未映射的日期！"
print(date_mapping["effective_report_date"].value_counts().sort("count", descending=True).head(10))

# ── 2. sentiment_df 打上 effective_report_date ─────────────────────────────
sentiment_with_eff = sentiment_df.join(date_mapping, on="date")

# ── 3. join ETF holdings ───────────────────────────────────────────────────
merged = sentiment_with_eff.join(
    etf_df.select(["factset_entity_id", "qid", "shares", "report_date"]),
    left_on=["qid", "effective_report_date"],
    right_on=["qid", "report_date"],
    how="inner",
)

# 验证
print(f"factset_entity_id nunique after join: {merged['factset_entity_id'].n_unique()}")

# ── 4. document 内部聚合：按 numOfSentences 加权不同 content_type ───────────
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
etf_signal = (
    doc_level
    .group_by(["factset_entity_id", "document_id", "date"])
    .agg(
        etf_sentiment=(
            (pl.col("doc_sentiment") * pl.col("shares")).sum()
            / pl.col("shares").sum()
        ),
        covered_shares=pl.col("shares").sum(),
    )
)

# ── 6. 可选：计算覆盖率用于质量过滤 ──────────────────────────────────────
total_shares = (
    etf_df
    .group_by(["factset_entity_id", "report_date"])
    .agg(total_shares=pl.col("shares").sum())
)

# 用 effective_report_date 关联 total_shares
etf_signal_with_coverage = (
    etf_signal
    .join(date_mapping, on="date")
    .join(
        total_shares,
        left_on=["factset_entity_id", "effective_report_date"],
        right_on=["factset_entity_id", "report_date"],
        how="left",
    )
    .with_columns(
        coverage_ratio=(pl.col("covered_shares") / pl.col("total_shares"))
    )
    .drop("effective_report_date")
)

print(etf_signal_with_coverage.head())
