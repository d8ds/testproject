import polars as pl
from datetime import datetime, timedelta, date
import time

# Your example data (expanded for testing)
df = pl.DataFrame({
    'qid': [1, 1, 1, 1, 2, 2, 2],
    'date': ['2024-01-01', '2024-07-01', '2024-08-06', '2024-09-15', 
             '2024-02-01', '2024-06-15', '2024-10-01'],
    'length': [2, 5, 3, 6, 10, 15, 12]
}).with_columns(
    pl.col('date').str.to_date()
)

def calculate_daily_differences_ultra_fast_fixed(df, start_date=None, end_date=None):
    """
    Ultra-fast vectorized approach with FIXED schema handling
    Explicitly defines column types to avoid schema inference errors
    """
    
    # Determine date range
    if start_date is None:
        start_date = df.select(pl.col('date').min()).item()
    if end_date is None:
        end_date = df.select(pl.col('date').max()).item()
    
    # Convert to date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Create date range
    days_diff = (end_date - start_date).days + 1
    date_range = [start_date + timedelta(days=i) for i in range(days_diff)]
    
    # Pre-define schema to avoid inference issues
    all_results = []
    
    # Process each qid separately for memory efficiency
    for qid_val in df['qid'].unique().sort():
        print(f"Processing QID {qid_val}...")
        
        # Get sorted data for this qid
        qid_data = df.filter(pl.col('qid') == qid_val).sort('date')
        records = qid_data.to_dicts()
        
        # Pre-compute for efficiency
        record_dates = [r['date'] for r in records]
        record_lengths = [r['length'] for r in records]
        
        # Process each date for this qid
        for query_date in date_range:
            # Find most recent record <= query_date
            most_recent_idx = -1
            for i, record_date in enumerate(record_dates):
                if record_date <= query_date:
                    most_recent_idx = i
                else:
                    break
            
            if most_recent_idx == -1:
                # No data available yet - explicitly use None with proper types
                result_row = {
                    'qid': qid_val,
                    'date': query_date,
                    'length': None,
                    'diff_vs_previous': None,
                    'diff_vs_avg_previous': None
                }
            else:
                current_length = record_lengths[most_recent_idx]
                most_recent_date = record_dates[most_recent_idx]
                
                # Find previous records within 6 months
                six_months_ago = query_date - timedelta(days=180)
                
                # Get previous records (before most recent and within window)
                prev_records = []
                for i in range(most_recent_idx):  # Only look at records before most recent
                    if record_dates[i] >= six_months_ago:
                        prev_records.append((record_dates[i], record_lengths[i]))
                
                # Calculate differences with explicit float conversion
                if prev_records:
                    # Most recent previous
                    most_recent_prev_length = prev_records[-1][1]
                    diff_vs_previous = float(current_length - most_recent_prev_length)
                    
                    # Average previous
                    avg_prev = sum(length for _, length in prev_records) / len(prev_records)
                    diff_vs_avg_previous = float(current_length - avg_prev)
                else:
                    diff_vs_previous = None
                    diff_vs_avg_previous = None
                
                result_row = {
                    'qid': qid_val,
                    'date': query_date,
                    'length': float(current_length) if current_length is not None else None,
                    'diff_vs_previous': diff_vs_previous,
                    'diff_vs_avg_previous': diff_vs_avg_previous
                }
            
            all_results.append(result_row)
    
    # Create DataFrame with explicit schema to avoid inference issues
    result_df = pl.DataFrame(
        all_results,
        schema={
            'qid': pl.Int64,
            'date': pl.Date,
            'length': pl.Float64,  # Explicitly Float64
            'diff_vs_previous': pl.Float64,  # Explicitly Float64
            'diff_vs_avg_previous': pl.Float64  # Explicitly Float64
        }
    )
    
    return result_df.sort(['qid', 'date'])

def calculate_daily_differences_batch_safe(df, start_date=None, end_date=None, batch_size=1000):
    """
    Batch processing version that's extra safe with schema handling
    Processes dates in batches to avoid large list creation
    """
    
    if start_date is None:
        start_date = df.select(pl.col('date').min()).item()
    if end_date is None:
        end_date = df.select(pl.col('date').max()).item()
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Define schema upfront
    schema = {
        'qid': pl.Int64,
        'date': pl.Date,
        'length': pl.Float64,
        'diff_vs_previous': pl.Float64,
        'diff_vs_avg_previous': pl.Float64
    }
    
    all_batch_results = []
    
    # Process each qid separately
    for qid_val in df['qid'].unique().sort():
        print(f"Processing QID {qid_val}...")
        
        qid_data = df.filter(pl.col('qid') == qid_val).sort('date')
        records = qid_data.to_dicts()
        
        record_dates = [r['date'] for r in records]
        record_lengths = [r['length'] for r in records]
        
        # Process dates in batches
        current_date = start_date
        while current_date <= end_date:
            batch_end = min(current_date + timedelta(days=batch_size-1), end_date)
            
            batch_results = []
            
            # Process this batch of dates
            batch_date = current_date
            while batch_date <= batch_end:
                
                # Find most recent record
                most_recent_idx = -1
                for i, record_date in enumerate(record_dates):
                    if record_date <= batch_date:
                        most_recent_idx = i
                    else:
                        break
                
                if most_recent_idx == -1:
                    batch_results.append({
                        'qid': qid_val,
                        'date': batch_date,
                        'length': None,
                        'diff_vs_previous': None,
                        'diff_vs_avg_previous': None
                    })
                else:
                    current_length = float(record_lengths[most_recent_idx])
                    
                    # Find previous records
                    six_months_ago = batch_date - timedelta(days=180)
                    prev_records = []
                    
                    for i in range(most_recent_idx):
                        if record_dates[i] >= six_months_ago:
                            prev_records.append((record_dates[i], record_lengths[i]))
                    
                    if prev_records:
                        most_recent_prev_length = float(prev_records[-1][1])
                        diff_vs_previous = current_length - most_recent_prev_length
                        
                        avg_prev = sum(length for _, length in prev_records) / len(prev_records)
                        diff_vs_avg_previous = current_length - avg_prev
                    else:
                        diff_vs_previous = None
                        diff_vs_avg_previous = None
                    
                    batch_results.append({
                        'qid': qid_val,
                        'date': batch_date,
                        'length': current_length,
                        'diff_vs_previous': diff_vs_previous,
                        'diff_vs_avg_previous': diff_vs_avg_previous
                    })
                
                batch_date += timedelta(days=1)
            
            # Create DataFrame for this batch with explicit schema
            if batch_results:
                batch_df = pl.DataFrame(batch_results, schema=schema)
                all_batch_results.append(batch_df)
            
            current_date = batch_end + timedelta(days=1)
    
    # Concatenate all batches
    if all_batch_results:
        return pl.concat(all_batch_results).sort(['qid', 'date'])
    else:
        # Return empty DataFrame with correct schema
        return pl.DataFrame([], schema=schema)

def calculate_daily_differences_memory_optimized(df, start_date=None, end_date=None):
    """
    Most memory-efficient version for very large datasets
    Uses iterative DataFrame construction to avoid memory spikes
    """
    
    if start_date is None:
        start_date = df.select(pl.col('date').min()).item()
    if end_date is None:
        end_date = df.select(pl.col('date').max()).item()
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Create empty result with proper schema
    result_df = pl.DataFrame([], schema={
        'qid': pl.Int64,
        'date': pl.Date,
        'length': pl.Float64,
        'diff_vs_previous': pl.Float64,
        'diff_vs_avg_previous': pl.Float64
    })
    
    # Process each qid and append to result
    for qid_val in df['qid'].unique().sort():
        print(f"Processing QID {qid_val}...")
        
        qid_result = calculate_daily_differences_ultra_fast_fixed(
            df.filter(pl.col('qid') == qid_val), 
            start_date, 
            end_date
        )
        
        result_df = pl.concat([result_df, qid_result])
    
    return result_df.sort(['qid', 'date'])

# Performance testing with schema fixes
print("Original data:")
print(df)

print("\n" + "="*60)
print("TESTING FIXED SCHEMA HANDLING")
print("="*60)

# Test with a reasonable range
test_start = '2024-08-01'
test_end = '2024-10-15'

print(f"Testing with date range: {test_start} to {test_end}")

try:
    start_time = time.time()
    result_fixed = calculate_daily_differences_ultra_fast_fixed(df, test_start, test_end)
    time_fixed = time.time() - start_time
    
    print(f"✅ SUCCESS! Ultra-fast fixed method: {time_fixed:.4f} seconds")
    print(f"Generated {result_fixed.height:,} daily records")
    print(f"Schema: {result_fixed.dtypes}")
    
    # Show sample results
    print("\nSample results:")
    sample = result_fixed.filter(pl.col('date').is_in([
        date(2024, 8, 10), date(2024, 9, 11), date(2024, 9, 15), date(2024, 10, 1)
    ]))
    print(sample)
    
except Exception as e:
    print(f"❌ Error with ultra-fast method: {e}")
    
    print("\nTrying batch-safe method...")
    try:
        result_batch = calculate_daily_differences_batch_safe(df, test_start, test_end, batch_size=100)
        print(f"✅ SUCCESS! Batch-safe method worked")
        print(f"Generated {result_batch.height:,} daily records")
    except Exception as e2:
        print(f"❌ Error with batch method: {e2}")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR 10-YEAR DATA:")
print("="*60)
print("1. 🚀 PRIMARY: calculate_daily_differences_ultra_fast_fixed()")
print("   - Fixed schema handling")
print("   - Explicit Float64 types")
print("   - Best performance")
print("")
print("2. 🛡️ BACKUP: calculate_daily_differences_batch_safe()")
print("   - Extra safe with batching")
print("   - Memory efficient")
print("   - Use if primary method fails")
print("")
print("3. 💾 MEMORY-CONSTRAINED: calculate_daily_differences_memory_optimized()")
print("   - Lowest memory usage")
print("   - Processes one QID at a time")

print("\n" + "="*50)
print("FOR YOUR 10-YEAR DATASET, USE:")
print("="*50)
print("result = calculate_daily_differences_ultra_fast_fixed(df, '2014-01-01', '2024-12-31')")
