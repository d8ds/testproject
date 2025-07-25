result = (
    df
    .lazy()
    .sort(["institution_id", "date"])
    .join(
        df.lazy(),
        on="institution_id",
        how="inner",
        suffix="_agg"
    )
    .filter(
        (pl.col("date_agg") <= pl.col("date")) &
        (pl.col("date_agg") >= pl.col("date") - pl.duration(days=180))
    )
    .group_by(["institution_id", "date"])
    .agg([
        pl.col("text").first().alias("current_text"),
        pl.col("text_agg").str.concat(delimiter=" ").alias("past_6m_text"),
        pl.col("date_agg").min().alias("earliest_date_in_window"),
        pl.col("date_agg").max().alias("latest_date_in_window"),
        pl.col("date_agg").count().alias("records_in_window")
    ])
    .collect()
)
