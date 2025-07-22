import polars as pl
from datetime import datetime, timedelta
import pandas as pd

# Sample data creation
def create_sample_data():
    """Create sample data for demonstration"""
    data = {
        'id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'A', 'B'],
        'doc_id': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10'],
        'date': [
            '2024-01-15', '2024-03-20', '2024-06-10', 
            '2024-02-05', '2024-07-12', 
            '2024-01-30', '2024-04-25', '2024-08-15',
            '2024-09-01', '2024-10-15'
        ],
        'doc_length': [100, 150, 200, 120, 180, 90, 160, 140, 110, 170]
    }
    
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col('date').str.to_date())
    return df

# Method 1: Using rolling aggregation with time window
def rolling_sum_method1(df, window_months=6):
    """
    Method 1: Using Polars' rolling aggregation with time window
    Most efficient for time-based windows
    """
    # Convert months to days (approximate)
    window_days = f"{window_months * 30}d"
    
    result = (
        df
        .sort(['id', 'date'])
        .with_columns([
            # Count documents in rolling window
            pl.col('doc_id').count().over(
                pl.col('id'), 
                order_by=pl.col('date'),
                mapping_strategy='join'
            ).rolling_count(window_size=window_days, by='date').alias('rolling_doc_count'),
            
            # Sum document lengths in rolling window  
            pl.col('doc_length').rolling_sum(
                window_size=window_days, 
                by='date'
            ).over('id').alias('rolling_length_sum')
        ])
    )
    
    return result

# Method 2: Using group_by_dynamic for precise month windows
def rolling_sum_method2(df, window_months=6):
    """
    Method 2: Using group_by_dynamic for more precise time windows
    Better for exact month calculations
    """
    # Create a complete date range for each ID
    date_range = df.select([
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date')
    ]).to_dicts()[0]
    
    # Generate all dates in range
    all_dates = pl.date_range(
        date_range['min_date'], 
        date_range['max_date'], 
        interval='1d',
        eager=True
    ).to_frame('date')
    
    # Get unique IDs
    unique_ids = df.select('id').unique()
    
    # Create complete grid
    complete_grid = unique_ids.join(all_dates, how='cross')
    
    # Join with original data
    expanded_df = complete_grid.join(df, on=['id', 'date'], how='left')
    
    # Calculate rolling sum
    result = (
        expanded_df
        .sort(['id', 'date'])
        .with_columns([
            pl.col('doc_id').fill_null('').alias('doc_id'),
            pl.col('doc_length').fill_null(0).alias('doc_length')
        ])
        .with_columns([
            # Rolling count of non-null documents
            (pl.col('doc_id') != '').cast(pl.Int32).rolling_sum(
                window_size=f"{window_months * 30}d", 
                by='date'
            ).over('id').alias('rolling_doc_count'),
            
            # Rolling sum of document lengths
            pl.col('doc_length').rolling_sum(
                window_size=f"{window_months * 30}d", 
                by='date'
            ).over('id').alias('rolling_length_sum')
        ])
        # Filter back to original dates with documents
        .filter(pl.col('doc_id') != '')
    )
    
    return result

# Method 3: Custom implementation with precise month calculation
def rolling_sum_method3(df, window_months=6):
    """
    Method 3: Custom implementation for exact month windows
    Most flexible but potentially slower for large datasets
    """
    def calculate_rolling_metrics(group_df):
        """Calculate rolling metrics for a single ID group"""
        group_df = group_df.sort('date')
        rolling_counts = []
        rolling_sums = []
        
        for i, row in enumerate(group_df.iter_rows(named=True)):
            current_date = row['date']
            # Calculate date 6 months ago
            start_date = current_date.replace(
                month=current_date.month - window_months if current_date.month > window_months 
                else current_date.month - window_months + 12,
                year=current_date.year if current_date.month > window_months 
                else current_date.year - 1
            )
            
            # Filter documents in window
            window_docs = group_df.filter(
                (pl.col('date') >= start_date) & (pl.col('date') <= current_date)
            )
            
            rolling_counts.append(len(window_docs))
            rolling_sums.append(window_docs.select(pl.col('doc_length').sum()).item())
        
        return group_df.with_columns([
            pl.Series('rolling_doc_count', rolling_counts),
            pl.Series('rolling_length_sum', rolling_sums)
        ])
    
    # Apply to each ID group
    result = (
        df
        .sort(['id', 'date'])
        .group_by('id', maintain_order=True)
        .map_groups(calculate_rolling_metrics)
    )
    
    return result

# Method 4: Using window functions with range-based window
def rolling_sum_method4(df, window_months=6):
    """
    Method 4: Using window functions with date range
    Good balance of performance and precision
    """
    # Convert to days (approximate)
    window_days = window_months * 30
    
    result = (
        df
        .sort(['id', 'date'])
        .with_columns([
            # Calculate days since epoch for each date
            (pl.col('date') - pl.date(1970, 1, 1)).dt.total_days().alias('days_since_epoch')
        ])
        .with_columns([
            # Rolling count using row-based window with date constraint
            pl.len().over(
                'id',
                order_by='days_since_epoch',
                mapping_strategy='join'
            ).alias('temp_count'),
            
            # Custom rolling calculation
            pl.struct(['date', 'doc_length']).map_elements(
                lambda x: 1,  # Count each document
                return_dtype=pl.Int32
            ).cumsum().over('id').alias('cumulative_count')
        ])
    )
    
    # For precise calculation, we need to use a different approach
    result = (
        df
        .sort(['id', 'date'])
        .with_columns([
            # Create a lagged date column for window start
            pl.col('date').shift(1).over('id').alias('prev_date')
        ])
        .with_columns([
            # Calculate rolling count for each row
            pl.struct(['id', 'date']).map_elements(
                lambda row: df.filter(
                    (pl.col('id') == row['id']) & 
                    (pl.col('date') <= row['date']) &
                    (pl.col('date') >= row['date'] - timedelta(days=window_months*30))
                ).height,
                return_dtype=pl.Int32
            ).alias('rolling_doc_count'),
            
            # Calculate rolling sum for each row  
            pl.struct(['id', 'date']).map_elements(
                lambda row: df.filter(
                    (pl.col('id') == row['id']) & 
                    (pl.col('date') <= row['date']) &
                    (pl.col('date') >= row['date'] - timedelta(days=window_months*30))
                ).select(pl.col('doc_length').sum()).item(),
                return_dtype=pl.Int64
            ).alias('rolling_length_sum')
        ])
        .drop('prev_date')
    )
    
    return result

# Optimized Method: Recommended approach
def rolling_sum_optimized(df, window_months=6):
    """
    Optimized method combining efficiency with accuracy
    Recommended for most use cases
    """
    window_duration = f"{window_months * 30}d"
    
    result = (
        df
        .sort(['id', 'date'])
        .group_by('id', maintain_order=True)
        .map_groups(
            lambda group: group.with_columns([
                # Rolling count of documents
                pl.lit(1).rolling_sum(
                    window_size=window_duration,
                    by='date'
                ).alias('rolling_doc_count'),
                
                # Rolling sum of document lengths
                pl.col('doc_length').rolling_sum(
                    window_size=window_duration,
                    by='date'
                ).alias('rolling_length_sum')
            ])
        )
    )
    
    return result

# Example usage and testing
def main():
    # Create sample data
    df = create_sample_data()
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Test different methods
    print("Method 1 - Basic rolling with time window:")
    result1 = rolling_sum_optimized(df, window_months=6)
    print(result1)
    print("\n" + "="*50 + "\n")
    
    # Performance comparison for larger datasets
    print("For production use, Method 'rolling_sum_optimized' is recommended")
    print("It provides the best balance of performance and accuracy")

if __name__ == "__main__":
    main()

# Additional utility functions
def validate_results(df, result_df):
    """Validate that rolling calculations are correct"""
    # Manual check for first few rows
    for row in result_df.head(3).iter_rows(named=True):
        id_val = row['id']
        date_val = row['date']
        window_start = date_val - timedelta(days=6*30)  # 6 months approximation
        
        manual_count = df.filter(
            (pl.col('id') == id_val) & 
            (pl.col('date') <= date_val) &
            (pl.col('date') >= window_start)
        ).height
        
        print(f"ID: {id_val}, Date: {date_val}")
        print(f"Calculated: {row['rolling_doc_count']}, Manual: {manual_count}")
        print(f"Match: {row['rolling_doc_count'] == manual_count}\n")

# Performance optimization tips:
"""
Performance Tips:
1. Use rolling_sum_optimized() for most cases
2. For very large datasets, consider:
   - Partitioning by ID if memory is limited
   - Using lazy evaluation: df.lazy().collect() 
   - Filtering date ranges before calculation
3. For exact month calculations (not 30-day approximation):
   - Use Method 3 but consider performance trade-offs
4. Index your data by ['id', 'date'] for faster operations
"""
