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
        .group_by('id', maintain_order=True)
        .map_groups(
            lambda group: group.with_columns([
                # Count documents in rolling window (using lit(1) to count each row)
                pl.lit(1).rolling_sum(
                    window_size=window_days, 
                    by='date'
                ).alias('rolling_doc_count'),
                
                # Sum document lengths in rolling window  
                pl.col('doc_length').rolling_sum(
                    window_size=window_days, 
                    by='date'
                ).alias('rolling_length_sum')
            ])
        )
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
    Method 4: Using join_asof for precise rolling calculations
    Good for exact time-based windows
    """
    window_days = window_months * 30
    
    # Create a helper function for rolling calculation
    def calculate_rolling_for_group(group_df):
        group_df = group_df.sort('date')
        results = []
        
        for i, current_row in enumerate(group_df.iter_rows(named=True)):
            current_date = current_row['date']
            start_date = current_date - timedelta(days=window_days)
            
            # Count docs in window
            window_data = group_df.filter(
                (pl.col('date') >= start_date) & 
                (pl.col('date') <= current_date)
            )
            
            doc_count = len(window_data)
            length_sum = window_data.select(pl.col('doc_length').sum()).item()
            
            results.append({
                'id': current_row['id'],
                'doc_id': current_row['doc_id'], 
                'date': current_row['date'],
                'doc_length': current_row['doc_length'],
                'rolling_doc_count': doc_count,
                'rolling_length_sum': length_sum
            })
        
        return pl.DataFrame(results)
    
    # Apply to each group
    result = (
        df
        .group_by('id')
        .map_groups(calculate_rolling_for_group)
    )
    
    return result

# Simple and working method - RECOMMENDED
def rolling_sum_simple(df, window_months=6):
    """
    Simple, working method for rolling calculations
    This is the most reliable approach
    """
    from datetime import timedelta
    
    def calculate_rolling_metrics(group_df):
        group_df = group_df.sort('date')
        rolling_counts = []
        rolling_sums = []
        
        for row in group_df.iter_rows(named=True):
            current_date = row['date']
            window_start = current_date - timedelta(days=window_months * 30)
            
            # Filter data within window
            window_mask = (
                (group_df['date'] >= window_start) & 
                (group_df['date'] <= current_date)
            )
            window_data = group_df.filter(window_mask)
            
            rolling_counts.append(len(window_data))
            rolling_sums.append(window_data.select(pl.col('doc_length').sum()).item())
        
        return group_df.with_columns([
            pl.Series('rolling_doc_count', rolling_counts),
            pl.Series('rolling_length_sum', rolling_sums)
        ])
    
    result = (
        df
        .sort(['id', 'date'])
        .group_by('id', maintain_order=True)
        .map_groups(calculate_rolling_metrics)
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
    print("Recommended Method - Simple and reliable:")
    result1 = rolling_sum_simple(df, window_months=6)
    print(result1)
    print("\n" + "="*30 + "\n")
    
    print("Method 1 - Using rolling_sum:")
    result2 = rolling_sum_method1(df, window_months=6)
    print(result2)
    print("\n" + "="*50 + "\n")
    
    # Performance comparison for larger datasets
    print("For production use:")
    print("- Use 'rolling_sum_simple' for reliability")
    print("- Use 'rolling_sum_method1' for better performance on large datasets")

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
