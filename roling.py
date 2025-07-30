import polars as pl
from datetime import datetime, timedelta

def simulate_online_rolling_stats(df, target_date, window_months=6, company_col="company", date_col="report_date", value_col="report_length"):
    """
    Simulate online setting by filtering data to only include records within the lookback window
    before calculating rolling statistics.
    
    Args:
        df: Polars DataFrame with company reports
        target_date: The target date for calculation (str or datetime)
        window_months: Number of months to look back (default: 6)
        company_col: Column name for company identifier
        date_col: Column name for report dates
        value_col: Column name for the metric to aggregate
    
    Returns:
        Polars DataFrame with rolling statistics for each company
    """
    
    # Convert target_date to datetime if it's a string
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d")
    
    # Calculate the start of the lookback window
    # Approximate months by using 30-day periods for simplicity
    # For more precise month calculations, you might want to use dateutil
    lookback_start = target_date - timedelta(days=window_months * 30)
    
    # Filter data to simulate online constraint
    online_data = df.filter(
        (pl.col(date_col) >= lookback_start) & 
        (pl.col(date_col) <= target_date)
    )
    
    # Calculate rolling statistics for each company
    result = online_data.group_by(company_col).agg([
        pl.col(value_col).sum().alias(f"total_{value_col}_6m"),
        pl.col(value_col).mean().alias(f"avg_{value_col}_6m"),
        pl.col(value_col).count().alias("report_count_6m"),
        pl.col(date_col).min().alias("earliest_report_date"),
        pl.col(date_col).max().alias("latest_report_date")
    ]).with_columns([
        pl.lit(target_date).alias("calculation_date"),
        pl.lit(lookback_start.date()).alias("window_start_date")
    ])
    
    return result

def batch_rolling_calculations(df, target_dates, window_months=6, company_col="company", date_col="report_date", value_col="report_length"):
    """
    Calculate rolling statistics for multiple target dates, simulating online constraints.
    
    Args:
        df: Polars DataFrame with company reports
        target_dates: List of target dates
        window_months: Number of months to look back
        company_col: Column name for company identifier
        date_col: Column name for report dates  
        value_col: Column name for the metric to aggregate
    
    Returns:
        Polars DataFrame with rolling statistics for all target dates
    """
    
    results = []
    
    for target_date in target_dates:
        daily_result = simulate_online_rolling_stats(
            df, target_date, window_months, company_col, date_col, value_col
        )
        results.append(daily_result)
    
    # Combine all results
    return pl.concat(results)

# Example usage
if __name__ == "__main__":
    # Create sample data - generate dates first, then create matching companies
    dates = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2025, 8, 1),
        interval="15d",
        eager=True
    )
    
    n_records = len(dates)
    companies = ["A", "B", "C"]
    
    sample_data = pl.DataFrame({
        "company": [companies[i % len(companies)] for i in range(n_records)],
        "report_date": dates,
        "report_length": [(100 + i * 10) % 300 + 50 for i in range(n_records)]  # Varying lengths
    })
    
    # Single target date calculation
    target_date = "2025-07-01"
    result_single = simulate_online_rolling_stats(
        sample_data, 
        target_date=target_date,
        window_months=6
    )
    
    print("Single target date result:")
    print(result_single)
    
    # Multiple target dates calculation
    target_dates = [
        datetime(2025, 6, 1),
        datetime(2025, 7, 1), 
        datetime(2025, 8, 1)
    ]
    
    result_batch = batch_rolling_calculations(
        sample_data,
        target_dates=target_dates,
        window_months=6
    )
    
    print("\nBatch calculation results:")
    print(result_batch.sort(["company", "calculation_date"]))

# Alternative approach using window functions for better performance with large datasets
def efficient_rolling_stats(df, target_dates, window_months=6, company_col="company", date_col="report_date", value_col="report_length"):
    """
    More efficient approach using Polars window functions for multiple target dates.
    """
    
    # Create a cross join with target dates
    target_df = pl.DataFrame({"calculation_date": target_dates})
    
    # Cross join to get all company-date combinations
    expanded = df.join(target_df, how="cross")
    
    # Calculate window start for each calculation date
    expanded = expanded.with_columns([
        (pl.col("calculation_date") - pl.duration(days=window_months * 30)).alias("window_start")
    ])
    
    # Filter to only include reports within the window
    windowed = expanded.filter(
        (pl.col(date_col) >= pl.col("window_start")) & 
        (pl.col(date_col) <= pl.col("calculation_date"))
    )
    
    # Calculate statistics
    result = windowed.group_by([company_col, "calculation_date"]).agg([
        pl.col(value_col).sum().alias(f"total_{value_col}_6m"),
        pl.col(value_col).mean().alias(f"avg_{value_col}_6m"),
        pl.col(value_col).count().alias("report_count_6m"),
        pl.col(date_col).min().alias("earliest_report_date"),
        pl.col(date_col).max().alias("latest_report_date")
    ])
    
    return result
