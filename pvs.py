import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_pca_signals(df, n_components=10, max_features=1000, batch_size=1000):
    """
    Generate PCA signals for each qid using text from last 6 months
    
    Parameters:
    - df: Polars DataFrame with columns [qid, document_id, sentence_id, filing_date, text]
    - n_components: Number of PCA components
    - max_features: Maximum features for TF-IDF vectorization
    - batch_size: Process qids in batches to manage memory
    """
    
    # Step 1: Prepare the data with proper date filtering
    df_prepared = (
        df
        .with_columns([
            pl.col("filing_date").str.to_date().alias("filing_date_parsed"),
            pl.col("qid").cast(pl.Utf8)  # Ensure qid is string for grouping
        ])
        #.with_columns([
            # Calculate max filing date for each qid
        #    pl.col("filing_date_parsed").max().over("qid").alias("max_filing_date")
        #])
        .with_columns([
            # Calculate 6 months before max filing date
            (pl.col("filing_date_parsed") - pl.duration(days=180)).alias("cutoff_date")
        ])
        .filter(
            # Keep only records within last 6 months for each qid
            pl.col("filing_date_parsed") >= pl.col("cutoff_date")
        )
        .sort(["qid", "filing_date_parsed", "document_id", "sentence_id"])
    )
    
    print(f"Data prepared. Shape before filtering: {df.shape}")
    print(f"Data prepared. Shape after filtering: {df_prepared.shape}")
    
    # Step 2: Get unique qids and process in batches
    unique_qids = df_prepared.select("qid").unique().to_series().to_list()
    total_qids = len(unique_qids)
    print(f"Processing {total_qids} unique qids in batches of {batch_size}")
    
    results = []
    
    for batch_start in range(0, total_qids, batch_size):
        batch_end = min(batch_start + batch_size, total_qids)
        batch_qids = unique_qids[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_qids-1)//batch_size + 1}: "
              f"qids {batch_start} to {batch_end-1}")
        
        # Process current batch
        batch_results = process_qid_batch(df_prepared, batch_qids, n_components, max_features)
        results.extend(batch_results)
    
    # Step 3: Combine results
    if results:
        final_df = pl.concat([pl.DataFrame(batch) for batch in results])
        return final_df
    else:
        return pl.DataFrame()

def process_qid_batch(df, qid_list, n_components, max_features):
    """Process a batch of qids for PCA signal generation"""
    
    batch_results = []
    
    # Filter data for current batch
    batch_df = df.filter(pl.col("qid").is_in(qid_list))
    
    for qid in qid_list:
        try:
            # Get text data for current qid
            qid_data = (
                batch_df
                .filter(pl.col("qid") == qid)
                .group_by(["document_id", "filing_date_parsed"])
                .agg([
                    pl.col("text").str.concat(" ").alias("document_text")
                ])
                .sort("filing_date_parsed")
            )
            
            if qid_data.height < 2:  # Need at least 2 documents for meaningful PCA
                print(f"Skipping qid {qid}: insufficient documents ({qid_data.height})")
                continue
            
            # Extract text and metadata
            texts = qid_data.select("document_text").to_series().to_list()
            filing_dates = qid_data.select("filing_date_parsed").to_series().to_list()
            doc_ids = qid_data.select("document_id").to_series().to_list()
            
            # Generate PCA signals
            pca_signals = calculate_pca_signals(texts, n_components, max_features)
            
            if pca_signals is not None:
                # Create result records
                for i, (doc_id, filing_date, signals) in enumerate(zip(doc_ids, filing_dates, pca_signals)):
                    result_record = {
                        'qid': qid,
                        'document_id': doc_id,
                        'filing_date': filing_date,
                        #'pca_signal_vector': signals.tolist(),  # Store as list for Polars
                    }
                    # Add individual PCA components as separate columns
                    for j, signal_val in enumerate(signals):
                        result_record[f'pca_component_{j+1}'] = float(signal_val)
                    
                    batch_results.append(result_record)
            
        except Exception as e:
            print(f"Error processing qid {qid}: {str(e)}")
            continue
    
    return batch_results

def calculate_pca_signals(texts, n_components, max_features):
    """Calculate PCA signals from text documents"""
    
    try:
        # Step 1: TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        if tfidf_matrix.shape[1] < n_components:
            n_components = min(tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1)
        
        if n_components <= 0:
            return None
        
        # Step 2: PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_signals = pca.fit_transform(tfidf_matrix.toarray())
        
        return pca_signals
        
    except Exception as e:
        print(f"Error in PCA calculation: {str(e)}")
        return None

def optimize_for_memory(df):
    """Optimize DataFrame for memory usage"""
    
    # Cast columns to appropriate types
    optimized_df = (
        df
        .with_columns([
            pl.col("qid").cast(pl.Utf8),
            pl.col("document_id").cast(pl.Utf8),
            pl.col("sentence_id").cast(pl.Utf8),
            pl.col("filing_date").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8)
        ])
    )
    
    return optimized_df

# Realistic data simulation for testing
def generate_large_test_dataset(n_qids=500, n_docs_per_qid=20, n_sentences_per_doc=15):
    """Generate a larger, more realistic dataset for testing"""
    
    import random
    from datetime import datetime, timedelta
    
    # Financial text templates for more realistic content
    financial_templates = [
        "Revenue increased by {percent}% compared to previous quarter due to {reason}",
        "Operating expenses were {trend} by {amount} million primarily from {category}",
        "Net income for the quarter was ${amount} million, {comparison} than expected",
        "Cash flow from operations {trend} to ${amount} million this period",
        "Total assets reached ${amount} billion, representing {growth}% growth",
        "Debt-to-equity ratio {trend} to {ratio}, indicating {assessment} financial position",
        "Market share in {sector} segment {trend} by {percent} percentage points",
        "Investment in {category} totaled ${amount} million during the quarter",
        "Risk-adjusted returns were {performance} expectations at {percent}%",
        "Regulatory compliance costs {trend} by ${amount} million year-over-year",
        "Customer acquisition costs {trend} while retention rates {improvement}",
        "Digital transformation initiatives generated ${amount} million in savings",
        "Supply chain disruptions resulted in {impact} of approximately ${amount} million",
        "ESG initiatives contributed to {metric} improvement of {percent}%",
        "Research and development spending {trend} to ${amount} million this quarter"
    ]
    
    # Words/phrases to fill templates
    fill_words = {
        'percent': [5, 8, 12, 15, 18, 22, 25, 30, 35, 40],
        'amount': [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000],
        'ratio': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        'reason': ['strong demand', 'market expansion', 'new product launches', 'cost optimization', 'strategic acquisitions'],
        'trend': ['increased', 'decreased', 'improved', 'declined', 'stabilized'],
        'comparison': ['higher', 'lower', 'consistent', 'better', 'worse'],
        'category': ['personnel', 'technology', 'marketing', 'operations', 'infrastructure'],
        'sector': ['retail', 'healthcare', 'technology', 'manufacturing', 'financial services'],
        'growth': [5, 8, 10, 12, 15, 18, 20, 25, 30],
        'assessment': ['strong', 'stable', 'improving', 'concerning', 'healthy'],
        'performance': ['above', 'below', 'meeting', 'exceeding'],
        'impact': ['an impact', 'losses', 'delays', 'additional costs'],
        'improvement': ['improved', 'remained stable', 'showed gains'],
        'metric': ['efficiency', 'sustainability', 'transparency', 'governance']
    }
    
    def generate_realistic_text():
        """Generate realistic financial text"""
        template = random.choice(financial_templates)
        filled_template = template
        
        for key, values in fill_words.items():
            if '{' + key + '}' in template:
                filled_template = filled_template.replace('{' + key + '}', str(random.choice(values)))
        
        return filled_template
    
    print(f"Generating dataset with {n_qids} qids, {n_docs_per_qid} docs per qid, {n_sentences_per_doc} sentences per doc...")
    print(f"Expected total records: {n_qids * n_docs_per_qid * n_sentences_per_doc:,}")
    
    data_records = []
    base_date = datetime(2023, 1, 1)
    
    for qid_idx in range(n_qids):
        qid = f"QID_{qid_idx:04d}"
        
        # Generate filing dates spanning 18 months (to test 6-month filtering)
        qid_dates = []
        for doc_idx in range(n_docs_per_qid):
            # Random date within 18 months, with some clustering
            days_offset = random.randint(0, 540)  # 18 months
            filing_date = base_date + timedelta(days=days_offset)
            qid_dates.append(filing_date.strftime('%Y-%m-%d'))
        
        # Create documents for this qid
        for doc_idx in range(n_docs_per_qid):
            doc_id = f"DOC_{qid_idx:04d}_{doc_idx:03d}"
            filing_date = qid_dates[doc_idx]
            
            # Generate sentences for this document
            for sent_idx in range(n_sentences_per_doc):
                sentence_id = sent_idx + 1
                text = generate_realistic_text()
                
                data_records.append({
                    'qid': qid,
                    'document_id': doc_id,
                    'sentence_id': sentence_id,
                    'filing_date': filing_date,
                    'text': text
                })
        
        # Progress indicator
        if (qid_idx + 1) % 50 == 0:
            print(f"Generated data for {qid_idx + 1}/{n_qids} qids...")
    
    print(f"Dataset generation complete. Total records: {len(data_records):,}")
    syn_data = pl.DataFrame(data_records)
    print(f"Data samples:\n {syn_data.head()}")
    return syn_data


def gen_sample():
    sample_data = {
        'qid': ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'] * 10,
        'document_id': ['D1', 'D1', 'D2', 'D3', 'D3', 'D4'] * 10,
        'sentence_id': list(range(60)),
        'filing_date': ['2024-01-15', '2024-01-15', '2024-02-20', '2024-03-10', '2024-03-10', '2024-04-05'] * 10,
        'text': [
            'Financial performance shows strong growth in revenue and profitability.',
            'Market conditions remain challenging but we see opportunities ahead.',
            'Investment in technology continues to drive operational efficiency.',
            'Customer satisfaction scores have improved significantly this quarter.',
            'Risk management frameworks have been strengthened across all divisions.',
            'Strategic partnerships are expanding our market reach effectively.'
        ] * 10
    }
    
    df = pl.DataFrame(sample_data)
    return df


# Comprehensive testing function
def run_comprehensive_test():
    """Run comprehensive tests with different dataset sizes"""
    
    print("=" * 80)
    print("COMPREHENSIVE PCA SIGNAL GENERATION TEST")
    print("=" * 80)
    
    # Test 1: Small dataset
    print("\n1. SMALL DATASET TEST (50 qids)")
    print("-" * 40)
    small_df = generate_large_test_dataset(n_qids=50, n_docs_per_qid=10, n_sentences_per_doc=8)
    small_df_opt = optimize_for_memory(small_df)
    
    print(f"Small dataset shape: {small_df_opt.shape}")
    print(f"Memory usage: ~{small_df_opt.estimated_size('mb'):.1f} MB")
    print(f"Small data sample\n {small_df_opt.head()}") 


    #small_df_opt = gen_sample()
    small_results = generate_pca_signals(
        small_df_opt, 
        n_components=2, 
        max_features=300,
        batch_size=25
    )
    
    if not small_results.is_empty():
        print(f"Small test results: {small_results.shape[0]} records generated")
        print("Sample of first result:")
        print(small_results.head(10))
    else:
        print('+' * 80)
    
    """ 
    # Test 2: Medium dataset
    print("\n2. MEDIUM DATASET TEST (200 qids)")
    print("-" * 40)
    medium_df = generate_large_test_dataset(n_qids=200, n_docs_per_qid=15, n_sentences_per_doc=12)
    medium_df_opt = optimize_for_memory(medium_df)
    
    print(f"Medium dataset shape: {medium_df_opt.shape}")
    print(f"Memory usage: ~{medium_df_opt.estimated_size('mb'):.1f} MB")
    print(f"Medium data sample\n {small_df_opt.head()}")
    
    medium_results = generate_pca_signals(
        medium_df_opt, 
        n_components=10, 
        max_features=500,
        batch_size=50
    )
    
    if not medium_results.is_empty():
        print(f"Medium test results: {medium_results.shape[0]} records generated")
        
        # Show some statistics
        pca_cols = [col for col in medium_results.columns if col.startswith('pca_component_')]
        if pca_cols:
            print(f"PCA Component Statistics (first 5 components):")
            stats = medium_results.select(pca_cols[:5]).describe()
            print(stats)
    
    # Test 3: Large dataset (memory permitting)
    print("\n3. LARGE DATASET TEST (500 qids)")
    print("-" * 40)
    try:
        large_df = generate_large_test_dataset(n_qids=500, n_docs_per_qid=20, n_sentences_per_doc=15)
        large_df_opt = optimize_for_memory(large_df)
        
        print(f"Large dataset shape: {large_df_opt.shape}")
        print(f"Memory usage: ~{large_df_opt.estimated_size('mb'):.1f} MB")
        print(f"Large data sample\n {small_df_opt.head()}")
        
        # Use smaller batch size for large dataset
        large_results = generate_pca_signals(
            large_df_opt, 
            n_components=2, 
            max_features=800,
            batch_size=25  # Smaller batches for large data
        )

        print(f"Large Results: \n {large_results.head()}")
        
        if not large_results.is_empty():
            print(f"Large test results: {large_results.shape[0]} records generated")
            
            # Analyze distribution of results by qid
            qid_counts = large_results.group_by("qid").agg(pl.count().alias("doc_count"))
            print(f"Documents per QID statistics:")
            print(qid_counts.select("doc_count").describe())
            
            # Check date filtering effectiveness
            date_range = large_results.select([
                pl.col("filing_date").min().alias("min_date"),
                pl.col("filing_date").max().alias("max_date")
            ])
            print(f"Date range in results: {date_range}")
        
    except MemoryError:
        print("Large dataset test skipped due to memory constraints")
    except Exception as e:
        print(f"Large dataset test failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
 """    
    return medium_results if 'medium_results' in locals() else small_results

# Updated main function
def main():
    """Main function with comprehensive testing"""
    return run_comprehensive_test()

if __name__ == "__main__":
    results = main()
    print('=' * 80)
    print(results.shape, results.unique(["qid", "filing_date"]).shape)
