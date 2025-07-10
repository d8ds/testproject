import polars as pl
import numpy as np

# --------------------------
# 1) Example: your base data
# --------------------------

docs = pl.DataFrame({
    "qid": ["A", "A", "A", "B", "B"],
    "filing_date": [
        "2021-01-15", "2021-03-10", "2021-05-25",
        "2021-02-01", "2021-02-15"
    ]
}).with_columns([
    pl.col("filing_date").str.to_datetime().alias("filing_date"),
    pl.col("filing_date").dt.truncate("1mo").alias("month")
])

print("\nDocs:")
print(docs)

# ----------------------------------------
# 2) Count filings per stock per calendar month
# ----------------------------------------

monthly = docs.groupby(["qid", "month"]).count().rename({"count": "n_filings"})
print("\nMonthly counts:")
print(monthly)

# ----------------------------------------
# 3) Build a full stock x month grid to fill missing months with zero
# ----------------------------------------

min_month = monthly["month"].min()
max_month = monthly["month"].max()

full_months = pl.date_range(min_month, max_month, interval="1mo", eager=True)
qids = monthly["qid"].unique()

full_grid = qids.join(full_months, how="cross").rename({"date": "month"})

monthly_full = full_grid.join(monthly, on=["qid", "month"], how="left").fill_null(0).sort(["qid", "month"])

print("\nMonthly grid (filled):")
print(monthly_full)

# ----------------------------------------
# 4) Rolling window using groupby_rolling (e.g., tr_
