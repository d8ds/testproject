import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def analyze_section_sentiment(df):
    """
    Analyze sentiment distribution across sections from 8K filings
    
    Args:
        df: polars DataFrame with columns ['date', 'document_id', 'section', 'sentiment_score']
    
    Returns:
        dict: comprehensive analysis results
    """
    
    # Basic statistics
    print("📊 Dataset Overview")
    print(f"Total records: {len(df):,}")
    print(f"Unique sections: {df['section'].n_unique()}")
    print(f"Unique documents: {df['document_id'].n_unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sentiment range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")
    print("-" * 50)
    
    # Section-level statistics
    section_stats = df.group_by('section').agg([
        pl.count('sentiment_score').alias('count'),
        pl.mean('sentiment_score').alias('mean_sentiment'),
        pl.median('sentiment_score').alias('median_sentiment'),
        pl.std('sentiment_score').alias('std_sentiment'),
        pl.min('sentiment_score').alias('min_sentiment'),
        pl.max('sentiment_score').alias('max_sentiment'),
        pl.quantile('sentiment_score', 0.25).alias('q25_sentiment'),
        pl.quantile('sentiment_score', 0.75).alias('q75_sentiment')
    ]).sort('mean_sentiment', descending=True)
    
    print("📋 Section Statistics (sorted by mean sentiment)")
    print(section_stats.to_pandas().round(3))
    print("-" * 50)
    
    # Overall statistics
    overall_stats = {
        'total_records': len(df),
        'total_sections': df['section'].n_unique(),
        'avg_sentiment': df['sentiment_score'].mean(),
        'median_sentiment': df['sentiment_score'].median(),
        'std_sentiment': df['sentiment_score'].std(),
        'positive_ratio': (df['sentiment_score'] > 0).sum() / len(df),
        'negative_ratio': (df['sentiment_score'] < 0).sum() / len(df),
        'neutral_ratio': (df['sentiment_score'] == 0).sum() / len(df)
    }
    
    print("🎯 Overall Sentiment Summary")
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value:,}")
    
    return section_stats, overall_stats

def create_sentiment_visualizations(df):
    """
    Create comprehensive sentiment visualizations
    
    Args:
        df: polars DataFrame with sentiment data
    """
    
    # Convert to pandas for easier plotting
    df_pd = df.to_pandas()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Box plot of sentiment by section
    plt.subplot(2, 3, 1)
    sections = df_pd['section'].unique()
    sentiment_by_section = [df_pd[df_pd['section'] == section]['sentiment_score'].values 
                           for section in sections]
    
    box_plot = plt.boxplot(sentiment_by_section, labels=sections, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sections)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Sentiment Distribution by Section (Box Plot)', fontsize=14, fontweight='bold')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 2. Bar chart of mean sentiment by section
    plt.subplot(2, 3, 2)
    section_means = df_pd.groupby('section')['sentiment_score'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(section_means)), section_means.values, 
                   color=[colors[i] for i in range(len(section_means))])
    
    # Color bars based on positive/negative sentiment
    for i, (bar, value) in enumerate(zip(bars, section_means.values)):
        if value >= 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    plt.title('Average Sentiment by Section', fontsize=14, fontweight='bold')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(range(len(section_means)), section_means.index, rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(section_means.values):
        plt.text(i, v + 0.01 if v >= 0 else v - 0.03, f'{v:.3f}', 
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    # 3. Histogram of overall sentiment distribution
    plt.subplot(2, 3, 3)
    plt.hist(df_pd['sentiment_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    plt.axvline(x=df_pd['sentiment_score'].mean(), color='orange', linestyle='-', linewidth=2, label='Mean')
    plt.axvline(x=df_pd['sentiment_score'].median(), color='green', linestyle='-', linewidth=2, label='Median')
    
    plt.title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Stacked histogram by section
    plt.subplot(2, 3, 4)
    sections_list = df_pd['section'].unique()
    sentiment_data = [df_pd[df_pd['section'] == section]['sentiment_score'].values 
                     for section in sections_list]
    
    plt.hist(sentiment_data, bins=20, alpha=0.7, label=sections_list, stacked=True)
    plt.title('Sentiment Distribution by Section (Stacked)', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 5. Scatter plot: Count vs Mean Sentiment
    plt.subplot(2, 3, 5)
    section_summary = df_pd.groupby('section').agg({
        'sentiment_score': ['count', 'mean', 'std']
    }).round(3)
    section_summary.columns = ['count', 'mean', 'std']
    
    scatter = plt.scatter(section_summary['mean'], section_summary['count'], 
                         s=section_summary['std']*500, alpha=0.6, c=section_summary['mean'], 
                         cmap='RdYlBu_r', edgecolors='black')
    
    # Add section labels
    for idx, section in enumerate(section_summary.index):
        plt.annotate(section, (section_summary.loc[section, 'mean'], 
                              section_summary.loc[section, 'count']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Section Analysis: Count vs Mean Sentiment\n(Bubble size = Std Dev)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Mean Sentiment Score')
    plt.ylabel('Number of Records')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(scatter, label='Mean Sentiment')
    plt.grid(True, alpha=0.3)
    
    # 6. Violin plot
    plt.subplot(2, 3, 6)
    section_data = []
    section_labels = []
    for section in sections_list:
        section_sentiments = df_pd[df_pd['section'] == section]['sentiment_score'].values
        if len(section_sentiments) > 1:  # Need at least 2 points for violin plot
            section_data.append(section_sentiments)
            section_labels.append(section)
    
    if section_data:
        parts = plt.violinplot(section_data, positions=range(len(section_data)), showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        plt.xticks(range(len(section_labels)), section_labels, rotation=45, ha='right')
        plt.title('Sentiment Distribution by Section (Violin Plot)', fontsize=14, fontweight='bold')
        plt.ylabel('Sentiment Score')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def sentiment_correlation_analysis(df):
    """
    Analyze correlations and patterns in sentiment data
    """
    print("\n🔍 Advanced Sentiment Analysis")
    print("=" * 50)
    
    # Convert date to datetime if it's not already
    df_analysis = df.with_columns([
        pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d').alias('date_parsed')
    ])
    
    # Time-based analysis
    monthly_sentiment = df_analysis.with_columns([
        pl.col('date_parsed').dt.strftime('%Y-%m').alias('month')
    ]).group_by('month').agg([
        pl.mean('sentiment_score').alias('avg_sentiment'),
        pl.count('sentiment_score').alias('count')
    ]).sort('month')
    
    print("📅 Monthly Sentiment Trends:")
    print(monthly_sentiment.to_pandas())
    
    # Section co-occurrence analysis
    print("\n📊 Section Co-occurrence in Documents:")
    doc_sections = df.group_by('document_id').agg([
        pl.col('section').unique().list().alias('sections'),
        pl.count('section').alias('section_count'),
        pl.mean('sentiment_score').alias('avg_doc_sentiment')
    ])
    
    print(f"Average sections per document: {doc_sections['section_count'].mean():.2f}")
    print(f"Documents with multiple sections: {(doc_sections['section_count'] > 1).sum()}")
    
    # Sentiment extremes
    print("\n🎯 Sentiment Extremes:")
    most_positive = df.filter(pl.col('sentiment_score') > 0.8)
    most_negative = df.filter(pl.col('sentiment_score') < -0.8)
    
    print(f"Highly positive records (>0.8): {len(most_positive)} ({len(most_positive)/len(df)*100:.1f}%)")
    print(f"Highly negative records (<-0.8): {len(most_negative)} ({len(most_negative)/len(df)*100:.1f}%)")
    
    if len(most_positive) > 0:
        print("\nMost positive sections:")
        print(most_positive.group_by('section').agg(pl.count()).sort('count', descending=True))
    
    if len(most_negative) > 0:
        print("\nMost negative sections:")
        print(most_negative.group_by('section').agg(pl.count()).sort('count', descending=True))

# Example usage function
def main():
    """
    Example usage with sample data
    """
    # Create sample data for demonstration
    np.random.seed(42)
    
    sections = ['Item 1.01', 'Item 2.02', 'Item 5.02', 'Item 7.01', 'Item 8.01']
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    
    sample_data = []
    for i in range(500):
        date = np.random.choice(dates)
        doc_id = f"DOC{i//3 + 1:03d}"  # Multiple sections per document
        section = np.random.choice(sections)
        
        # Create realistic sentiment patterns
        if section == 'Item 1.01':  # Material agreements - slightly positive
            sentiment = np.random.normal(0.1, 0.3)
        elif section == 'Item 2.02':  # Financial results - more variable
            sentiment = np.random.normal(0.05, 0.4)
        elif section == 'Item 5.02':  # Leadership changes - neutral to negative
            sentiment = np.random.normal(-0.05, 0.25)
        elif section == 'Item 7.01':  # Regulatory matters - slightly negative
            sentiment = np.random.normal(-0.1, 0.3)
        else:  # Other events
            sentiment = np.random.normal(0, 0.35)
        
        # Clip to valid range
        sentiment = np.clip(sentiment, -1, 1)
        
        sample_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'document_id': doc_id,
            'section': section,
            'sentiment_score': sentiment
        })
    
    # Create Polars DataFrame
    df = pl.DataFrame(sample_data)
    
    print("🚀 Running 8K Filing Section Sentiment Analysis")
    print("=" * 60)
    
    # Run analysis
    section_stats, overall_stats = analyze_section_sentiment(df)
    create_sentiment_visualizations(df)
    sentiment_correlation_analysis(df)
    
    return df, section_stats, overall_stats

# Function to use with your actual data
def analyze_your_data(df):
    """
    Use this function with your actual Polars DataFrame
    
    Args:
        df: Your Polars DataFrame with columns ['date', 'document_id', 'section', 'sentiment_score']
    """
    # Validate input
    required_columns = ['date', 'document_id', 'section', 'sentiment_score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data types and ranges
    if not df['sentiment_score'].dtype.is_numeric():
        raise ValueError("sentiment_score must be numeric")
    
    if df['sentiment_score'].min() < -1 or df['sentiment_score'].max() > 1:
        print("⚠️  Warning: sentiment_score values outside [-1, 1] range detected")
    
    # Run the analysis
    print("🚀 Analyzing Your 8K Filing Sentiment Data")
    print("=" * 60)
    
    section_stats, overall_stats = analyze_section_sentiment(df)
    create_sentiment_visualizations(df)
    sentiment_correlation_analysis(df)
    
    return section_stats, overall_stats

if __name__ == "__main__":
    # Run example with sample data
    df, section_stats, overall_stats = main()
