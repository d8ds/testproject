import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def collect_rolling_sections_method1(df):
    """
    Method 1: Using join_asof with filtering
    Most efficient for large datasets
    
    Args:
        df: Polars DataFrame with columns ['id', 'date', 'section']
    
    Returns:
        DataFrame with rolling 6-month section collections
    """
    
    # Ensure date is proper date type
    df = df.with_columns([
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d').alias('date')
    ])
    
    # Create all unique combinations of id and date for business days
    # Get date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # Create business day range
    business_days = pl.DataFrame({
        'date': pl.date_range(min_date, max_date, interval='1d', eager=True)
    }).filter(
        pl.col('date').dt.weekday() <= 5  # Monday=1, Friday=5
    )
    
    # Get all unique IDs
    unique_ids = df.select('id').unique()
    
    # Cross join to get all id-date combinations
    all_combinations = unique_ids.join(business_days, how='cross')
    
    # For each combination, collect sections from past 6 months
    result = all_combinations.with_columns([
        (pl.col('date') - pl.duration(days=180)).alias('start_date')
    ]).join(
        df.select(['id', 'date', 'section']),
        on='id',
        how='left'
    ).filter(
        (pl.col('date_right') >= pl.col('start_date')) & 
        (pl.col('date_right') <= pl.col('date'))
    ).group_by(['id', 'date']).agg([
        pl.col('section').unique().drop_nulls().alias('sections_past_6m')
    ]).sort(['id', 'date'])
    
    return result

def collect_rolling_sections_method2(df):
    """
    Method 2: Using group_by with rolling window logic
    More straightforward but potentially slower
    """
    
    # Ensure date is proper date type
    df = df.with_columns([
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d').alias('date')
    ])
    
    # Sort by id and date
    df = df.sort(['id', 'date'])
    
    # For each row, collect sections from past 6 months
    result = df.with_columns([
        (pl.col('date') - pl.duration(days=180)).alias('start_date')
    ]).join(
        df.select(['id', 'date', 'section']).rename({'date': 'ref_date', 'section': 'ref_section'}),
        on='id',
        how='left'
    ).filter(
        (pl.col('ref_date') >= pl.col('start_date')) & 
        (pl.col('ref_date') <= pl.col('date'))
    ).group_by(['id', 'date']).agg([
        pl.col('ref_section').unique().alias('sections_past_6m')
    ]).sort(['id', 'date'])
    
    return result

def collect_rolling_sections_optimized(df):
    """
    Method 3: Optimized version with business day filtering
    Recommended for production use
    """
    
    # Ensure date is proper date type and sort
    df = df.with_columns([
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d').alias('date')
    ]).sort(['id', 'date'])
    
    # Create a helper function to generate business days
    def get_business_days(start_date, end_date):
        """Generate business days between start and end date"""
        business_days = pl.date_range(start_date, end_date, interval='1d', eager=True)
        business_days_df = pl.DataFrame({'date': business_days})
        return business_days_df.filter(
            pl.col('date').dt.weekday() <= 5  # Filter weekends
        )['date'].to_list()
    
    # Get unique IDs and date range
    unique_ids = df['id'].unique().to_list()
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # Generate all business days
    business_days = get_business_days(min_date, max_date)
    
    # Create the cross product of IDs and business days
    all_combinations = pl.DataFrame({
        'id': [id_val for id_val in unique_ids for _ in business_days],
        'date': business_days * len(unique_ids)
    })
    
    # Join with original data to get sections for past 6 months
    result = all_combinations.join(
        df.with_columns([
            pl.col('date').alias('section_date'),
            pl.col('section').alias('section_name')
        ]).select(['id', 'section_date', 'section_name']),
        on='id',
        how='left'
    ).filter(
        # Keep sections from past 6 months (180 days)
        (pl.col('section_date') >= (pl.col('date') - pl.duration(days=180))) &
        (pl.col('section_date') <= pl.col('date'))
    ).group_by(['id', 'date']).agg([
        pl.col('section_name').unique().drop_nulls().alias('sections_past_6m'),
        pl.col('section_name').count().alias('total_sections_count'),
        pl.col('section_name').unique().drop_nulls().len().alias('unique_sections_count')
    ]).sort(['id', 'date'])
    
    return result

def collect_rolling_sections_pandas_style(df):
    """
    Method 4: Using pandas-style rolling window
    For those familiar with pandas approach
    """
    
    # Convert to pandas for rolling operations
    df_pd = df.to_pandas()
    df_pd['date'] = pd.to_datetime(df_pd['date'])
    
    # Create business day range
    min_date = df_pd['date'].min()
    max_date = df_pd['date'].max()
    business_days = pd.bdate_range(start=min_date, end=max_date)
    
    # Create all combinations
    unique_ids = df_pd['id'].unique()
    all_combinations = pd.MultiIndex.from_product([unique_ids, business_days], 
                                                  names=['id', 'date']).to_frame(index=False)
    
    result_list = []
    
    for (id_val, date_val) in zip(all_combinations['id'], all_combinations['date']):
        # Get sections from past 6 months
        six_months_ago = date_val - pd.DateOffset(months=6)
        
        sections = df_pd[
            (df_pd['id'] == id_val) & 
            (df_pd['date'] >= six_months_ago) & 
            (df_pd['date'] <= date_val)
        ]['section'].unique().tolist()
        
        result_list.append({
            'id': id_val,
            'date': date_val,
            'sections_past_6m': sections,
            'unique_sections_count': len(sections)
        })
    
    # Convert back to Polars
    result_df = pl.DataFrame(result_list)
    
    return result_df

def collect_rolling_sections_memory_efficient(df):
    """
    Method 5: Memory-efficient version for very large datasets
    Process in chunks to avoid memory issues
    """
    
    # Ensure date is proper date type
    df = df.with_columns([
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d').alias('date')
    ]).sort(['id', 'date'])
    
    # Process by ID to reduce memory usage
    unique_ids = df['id'].unique().to_list()
    
    results = []
    
    for id_val in unique_ids:
        # Get data for this ID
        id_data = df.filter(pl.col('id') == id_val)
        
        # Get date range for this ID
        min_date = id_data['date'].min()
        max_date = id_data['date'].max()
        
        # Generate business days for this ID
        business_days = pl.date_range(min_date, max_date, interval='1d', eager=True)
        business_days_df = pl.DataFrame({'date': business_days}).filter(
            pl.col('date').dt.weekday() <= 5
        )
        
        # For each business day, collect past 6 months sections
        id_result = business_days_df.with_columns([
            pl.lit(id_val).alias('id')
        ]).join(
            id_data.with_columns([
                pl.col('date').alias('section_date'),
                pl.col('section').alias('section_name')
            ]).select(['section_date', 'section_name']),
            how='cross'
        ).filter(
            (pl.col('section_date') >= (pl.col('date') - pl.duration(days=180))) &
            (pl.col('section_date') <= pl.col('date'))
        ).group_by(['id', 'date']).agg([
            pl.col('section_name').unique().drop_nulls().alias('sections_past_6m'),
            pl.col('section_name').unique().drop_nulls().len().alias('unique_sections_count')
        ])
        
        results.append(id_result)
    
    # Combine results
    final_result = pl.concat(results).sort(['id', 'date'])
    
    return final_result

# Example usage and testing
def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Sample data
    ids = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    sections = ['Item 1.01', 'Item 2.02', 'Item 5.02', 'Item 7.01', 'Item 8.01']
    
    # Generate random dates over 1 year
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    sample_data = []
    
    for _ in range(1000):
        # Random date
        random_date = start_date + timedelta(
            days=np.random.randint(0, (end_date - start_date).days)
        )
        
        # Skip weekends
        if random_date.weekday() >= 5:
            continue
            
        sample_data.append({
            'id': np.random.choice(ids),
            'date': random_date.strftime('%Y-%m-%d'),
            'section': np.random.choice(sections)
        })
    
    return pl.DataFrame(sample_data)

def benchmark_methods(df):
    """Benchmark different methods"""
    import time
    
    methods = [
        ('Optimized Method', collect_rolling_sections_optimized),
        ('Method 1 (join_asof)', collect_rolling_sections_method1),
        ('Method 2 (group_by)', collect_rolling_sections_method2),
        ('Memory Efficient', collect_rolling_sections_memory_efficient)
    ]
    
    results = {}
    
    for name, method in methods:
        print(f"\nTesting {name}...")
        start_time = time.time()
        
        try:
            result = method(df.clone())
            end_time = time.time()
            
            print(f"✅ {name}: {end_time - start_time:.2f}s")
            print(f"   Result shape: {result.shape}")
            print(f"   Sample result:")
            print(result.head(3))
            
            results[name] = {
                'time': end_time - start_time,
                'result': result
            }
            
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
    
    return results

# Main function for demonstration
def main():
    """Demonstrate the functionality"""
    
    print("🚀 Creating sample data...")
    df = create_sample_data()
    
    print(f"📊 Sample data shape: {df.shape}")
    print("Sample data:")
    print(df.head(10))
    
    print("\n" + "="*60)
    print("🔍 Running benchmark of different methods...")
    
    # Use a smaller subset for benchmarking
    small_df = df.head(200)
    results = benchmark_methods(small_df)
    
    print("\n" + "="*60)
    print("🎯 Recommended usage:")
    print("""
# For most use cases, use the optimized method:
result = collect_rolling_sections_optimized(your_df)

# For very large datasets, use memory-efficient method:
result = collect_rolling_sections_memory_efficient(your_df)

# For pandas users familiar with rolling windows:
result = collect_rolling_sections_pandas_style(your_df)
""")
    
    return df, results

if __name__ == "__main__":
    df, results = main()
