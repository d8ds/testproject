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
# ── 1. 改为 per-qid asof join ──────────────────────────────────────────────
# 左表：sentiment 里每个 qid + date 的唯一组合
sentiment_dates = (
    sentiment_df.select(["qid", "date"])
    .unique()
    .sort(["qid", "date"])
)

# 右表：etf_df 里每个 qid 实际出现的 report_date
etf_dates = (
    etf_df.select(["qid", "report_date"])
    .unique()
    .sort(["qid", "report_date"])
    .rename({"report_date": "effective_report_date"})
)

# per-qid backward：找该 qid 最近的历史持仓快照
qid_date_mapping_bwd = sentiment_dates.join_asof(
    etf_dates,
    left_on="date",
    right_on="effective_report_date",
    by="qid",
    strategy="backward",
)

# per-qid forward：对 null 的找该 qid 最近的未来持仓快照
qid_date_mapping_fwd = sentiment_dates.join_asof(
    etf_dates,
    left_on="date",
    right_on="effective_report_date",
    by="qid",
    strategy="forward",
)

# 合并
qid_date_mapping = qid_date_mapping_bwd.with_columns(
    pl.when(pl.col("effective_report_date").is_null())
    .then(qid_date_mapping_fwd["effective_report_date"])
    .otherwise(pl.col("effective_report_date"))
    .alias("effective_report_date")
)

# 仍然为 null 的说明这个 qid 根本不在任何 ETF 持仓里，后续 inner join 会自然过滤
print(f"qid_date_mapping null count: {qid_date_mapping['effective_report_date'].null_count()}")

# ── 2. sentiment_df 打上 effective_report_date ─────────────────────────────
sentiment_with_eff = sentiment_df.join(qid_date_mapping, on=["qid", "date"])

# ── 3. join ETF holdings（现在 effective_report_date 是 per-qid 的）─────────
merged = sentiment_with_eff.join(
    etf_df.select(["factset_entity_id", "qid", "shares", "report_date"]),
    left_on=["qid", "effective_report_date"],
    right_on=["qid", "report_date"],
    how="inner",
)

print(f"factset_entity_id nunique: {merged['factset_entity_id'].n_unique()}")
