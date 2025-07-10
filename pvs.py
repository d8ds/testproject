import polars as pl
import numpy as np
from scipy.stats import entropy

def calculate_temporal_entropy(df, qid, window_months=12):
    """
    Calculate entropy of filing frequency over time windows
    """
    # Group by qid and month
    monthly_filings = (
        df.filter(pl.col('qid') == qid)
        .with_columns([
            pl.col('filing_date').dt.truncate('1mo').alias('month')
        ])
        .group_by(['qid', 'month'])
        .agg([
            pl.col('document_id').n_unique().alias('filing_count')
        ])
        .sort('month')
    )
    
    # Calculate rolling entropy
    filing_counts = monthly_filings['filing_count'].to_numpy()
    
    # Normalize to probability distribution
    if filing_counts.sum() > 0:
        probs = filing_counts / filing_counts.sum()
        return entropy(probs + 1e-10)  # Add small epsilon for numerical stability
    return 0

def calculate_burst_entropy(df, lookback_months=6):
    """
    Capture entropy of filing bursts vs normal periods
    """
    result = (
        df.with_columns([
            pl.col('filing_date').dt.truncate('1mo').alias('month')
        ])
        .group_by(['qid', 'month'])
        .agg([
            pl.col('document_id').n_unique().alias('monthly_filings')
        ])
        .sort(['qid', 'month'])
        .with_columns([
            # Calculate rolling statistics
            pl.col('monthly_filings').rolling_mean(window_size=lookback_months).over('qid').alias('avg_filings'),
            pl.col('monthly_filings').rolling_std(window_size=lookback_months).over('qid').alias('std_filings')
        ])
        .with_columns([
            # Identify bursts (filings > mean + 2*std)
            (pl.col('monthly_filings') > pl.col('avg_filings') + 2 * pl.col('std_filings')).alias('is_burst'),
            # Normal periods
            (pl.col('monthly_filings') <= pl.col('avg_filings') + pl.col('std_filings')).alias('is_normal')
        ])
        .group_by('qid')
        .agg([
            pl.col('is_burst').sum().alias('burst_months'),
            pl.col('is_normal').sum().alias('normal_months'),
            pl.col('monthly_filings').count().alias('total_months')
        ])
        .with_columns([
            # Calculate burst entropy
            (pl.col('burst_months') / pl.col('total_months')).alias('burst_prob'),
            (pl.col('normal_months') / pl.col('total_months')).alias('normal_prob')
        ])
    )
    
    return result

def calculate_multiscale_entropy(df, scales=[1, 3, 6, 12]):
    """
    Calculate entropy at multiple time scales
    """
    features = []
    
    for scale in scales:
        scale_entropy = (
            df.with_columns([
                pl.col('filing_date').dt.truncate(f'{scale}mo').alias(f'period_{scale}mo')
            ])
            .group_by(['qid', f'period_{scale}mo'])
            .agg([
                pl.col('document_id').n_unique().alias('filing_count')
            ])
            .group_by('qid')
            .agg([
                pl.col('filing_count').map_elements(
                    lambda x: entropy(x / x.sum() + 1e-10) if x.sum() > 0 else 0
                ).alias(f'entropy_{scale}mo')
            ])
        )
        features.append(scale_entropy)
    
    # Join all scales
    result = features[0]
    for feat in features[1:]:
        result = result.join(feat, on='qid')
    
    return result


def calculate_relative_entropy(df, sector_mapping):
    """
    Calculate entropy relative to sector and market norms
    """
    # Add sector information
    df_with_sector = df.join(sector_mapping, on='qid')
    
    # Calculate company entropy
    company_entropy = calculate_temporal_entropy(df_with_sector)
    
    # Calculate sector median entropy
    sector_entropy = (
        company_entropy.group_by('sector')
        .agg([
            pl.col('entropy').median().alias('sector_median_entropy')
        ])
    )
    
    # Calculate relative measures
    result = (
        company_entropy.join(sector_entropy, on='sector')
        .with_columns([
            (pl.col('entropy') / pl.col('sector_median_entropy')).alias('relative_entropy'),
            (pl.col('entropy') - pl.col('sector_median_entropy')).alias('excess_entropy')
        ])
    )
    
    return result

def generate_entropy_features(df):
    """
    Generate comprehensive entropy-based features
    """
    features = []
    
    # 1. Basic temporal entropy
    temporal_entropy = calculate_multiscale_entropy(df)
    features.append(temporal_entropy)
    
    # 2. Burst entropy
    burst_entropy = calculate_burst_entropy(df)
    features.append(burst_entropy)
    
    # 3. Consistency entropy (inverse of coefficient of variation)
    consistency = (
        df.with_columns([
            pl.col('filing_date').dt.truncate('1mo').alias('month')
        ])
        .group_by(['qid', 'month'])
        .agg([
            pl.col('document_id').n_unique().alias('monthly_filings')
        ])
        .group_by('qid')
        .agg([
            pl.col('monthly_filings').mean().alias('mean_filings'),
            pl.col('monthly_filings').std().alias('std_filings')
        ])
        .with_columns([
            (pl.col('std_filings') / pl.col('mean_filings')).alias('cv_filings'),
            (1 / (pl.col('std_filings') / pl.col('mean_filings'))).alias('consistency_score')
        ])
    )
    features.append(consistency)
    
    # 4. Recent vs historical entropy
    recent_entropy = calculate_temporal_entropy(
        df.filter(pl.col('filing_date') >= pl.col('filing_date').max() - pl.duration(months=6))
    )
    historical_entropy = calculate_temporal_entropy(
        df.filter(pl.col('filing_date') < pl.col('filing_date').max() - pl.duration(months=6))
    )
    
    entropy_change = recent_entropy.join(historical_entropy, on='qid', suffix='_historical')
    
    # Combine all features
    final_features = features[0]
    for feat in features[1:]:
        final_features = final_features.join(feat, on='qid', how='outer')
    
    return final_features


