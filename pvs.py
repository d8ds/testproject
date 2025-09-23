def pure_polars_rolling(df):
    """
    Pure Polars solution that works with large datasets
    """
    # First, get all the non-null data
    clean_df = (
        df
        .filter(pl.col('section').is_not_null())
        .sort(['qid', 'date'])
    )
    
    # Create a helper function for each group
    def rolling_agg(group_df):
        dates = group_df.get_column('date').to_list()
        sections = group_df.get_column('section').to_list()
        
        result_sections = []
        for i, current_date in enumerate(dates):
            start_date = current_date - pl.duration(days=180)
            
            # Get all sections within the window
            window_sections = [
                sections[j] for j, date in enumerate(dates)
                if start_date <= date <= current_date
            ]
            result_sections.append(window_sections)
        
        return group_df.with_columns([
            pl.Series('sections_180d', result_sections)
        ])
    
    # Process by qid groups
    return (
        clean_df
        .group_by('qid', maintain_order=True)
        .map_groups(rolling_agg)
    )


def chunked_rolling_sections(df, chunk_size=10000):
    """
    Process in chunks for very large datasets
    """
    clean_df = df.filter(pl.col('section').is_not_null()).sort(['qid', 'date'])
    unique_qids = clean_df.get_column('qid').unique().to_list()
    
    results = []
    
    # Process qids in chunks
    for i in range(0, len(unique_qids), chunk_size):
        chunk_qids = unique_qids[i:i + chunk_size]
        
        chunk_result = sql_style_rolling(
            clean_df.filter(pl.col('qid').is_in(chunk_qids))
        )
        
        results.append(chunk_result)
        
        # Optional: print progress
        print(f"Processed {min(i + chunk_size, len(unique_qids))} / {len(unique_qids)} qids")
    
    return pl.concat(results)


def sql_style_rolling(df):
    """
    SQL-style approach that works reliably
    """
    clean_df = df.filter(pl.col('section').is_not_null())
    
    return (
        clean_df
        .select(['qid', 'date'])
        .unique()
        .sort(['qid', 'date'])
        .join(
            clean_df.select(['qid', 'date', 'section']),
            on='qid',
            how='left',
            suffix='_section'
        )
        .filter(
            (pl.col('date_section') <= pl.col('date')) &
            (pl.col('date_section') >= pl.col('date') - pl.duration(days=180))
        )
        .group_by(['qid', 'date'])
        .agg([
            pl.col('section').drop_nulls().alias('sections_180d')
        ])
        .sort(['qid', 'date'])
    )

import polars as pl
from datetime import timedelta

# Ensure date column is properly typed
df = df.with_columns(pl.col("date").cast(pl.Date))

# Sort by qid and date for efficient window operations
df = df.sort(["qid", "date"])

# Create rolling aggregation
result = df.with_columns(
    pl.col("section")
    .filter(pl.col("section").is_not_null())  # Only non-null sections
    .list.eval(pl.element().drop_nulls())     # Remove nulls from each group
    .over(
        "qid", 
        order_by="date",
        mapping_strategy="group_to_rows"
    )
    .rolling_map(
        function=lambda s: s.explode().unique().sort(),
        window_size=timedelta(days=181),  # 180 days + current day
        min_periods=1
    )
    .alias("sections_180d")
)
