import polars as pl
from datetime import timedelta

# Assuming your dataframe is called 'df'
result = (
    df
    .sort(['id', 'date'])
    .group_by('id')
    .map_groups(
        lambda group: group.with_columns([
            pl.col('section')
            .rolling_map(
                function=lambda s: [x for x in s if x is not None],
                window_size=timedelta(days=181),  # 181 to include current day
                min_periods=1
            )
            .alias('sections_180d')
        ])
    )
)
#--------------
# Create a helper function to get date ranges
def get_rolling_sections(df):
    return (
        df
        .sort(['id', 'date'])
        .with_columns([
            (pl.col('date') - pl.duration(days=180)).alias('start_date')
        ])
        .join_asof(
            df.filter(pl.col('section').is_not_null())
              .sort(['id', 'date']),
            left_on=['id', 'date'],
            right_on=['id', 'date'],
            strategy='backward'
        )
        .group_by(['id', 'date', 'start_date'])
        .agg([
            pl.col('section').filter(
                pl.col('date').is_between(pl.col('start_date'), pl.col('date'))
            ).drop_nulls().alias('sections_180d')
        ])
    )
#---------------
import polars as pl
from datetime import timedelta

def efficient_rolling_sections(df):
    return (
        df.lazy()  # Use lazy evaluation
        .filter(pl.col('section').is_not_null())  # Filter nulls early
        .sort(['id', 'date'])  # Sort once
        .group_by('id', maintain_order=True)  # Maintain order for efficiency
        .map_groups(
            lambda group: group.with_columns([
                pl.col('section')
                .rolling_map(
                    function=lambda s: s.to_list(),  # More efficient than list comprehension
                    window_size=timedelta(days=181),
                    min_periods=1,
                    closed='both'
                )
                .alias('sections_180d')
            ])
        )
        .collect(streaming=True)  # Use streaming for memory efficiency
    )


def highly_optimized_rolling_sections(df):
    """
    Most efficient approach for large datasets
    """
    return (
        df.lazy()
        .with_columns([
            pl.col('date').cast(pl.Date),  # Ensure proper date type
        ])
        .filter(pl.col('section').is_not_null())  # Remove nulls early
        .sort(['id', 'date'])
        .with_columns([
            # Create a more efficient rolling aggregation
            pl.col('section')
            .rolling_list(
                window_size=timedelta(days=181),
                min_periods=1,
                closed='both'
            )
            .over('id', order_by='date')  # Use over instead of group_by when possible
            .alias('sections_180d')
        ])
        .collect(streaming=True, slice_pushdown=True)
    )

def chunked_rolling_sections(df, chunk_size=1_000_000):
    """
    Process in chunks for datasets that don't fit in memory
    """
    # Get unique qids to avoid splitting groups
    unique_qids = df.select('id').unique().sort('id')
    
    results = []
    
    for i in range(0, len(unique_qids), chunk_size):
        chunk_qids = unique_qids.slice(i, chunk_size)
        
        chunk_result = (
            df.lazy()
            .filter(pl.col('id').is_in(chunk_qids.get_column('id')))
            .pipe(highly_optimized_rolling_sections)  # Apply your function
        )
        
        results.append(chunk_result)
    
    return pl.concat(results)

# For datasets > 10GB, consider this approach
def memory_efficient_approach(df):
    return (
        df.lazy()
        .with_columns([
            pl.col('date').cast(pl.Date)
        ])
        # Only select necessary columns early
        .select(['qid', 'date', 'section'])
        .filter(pl.col('section').is_not_null())
        .sort(['qid', 'date'])
        .group_by('qid', maintain_order=True)
        .map_groups(
            lambda group: group.with_columns([
                pl.col('section')
                .rolling_list(window_size='180d', min_periods=1)
                .alias('sections_180d')
            ])
        )
        .collect(
            streaming=True,
            slice_pushdown=True,
            predicate_pushdown=True,
            projection_pushdown=True
        )
    )
