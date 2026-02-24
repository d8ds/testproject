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
