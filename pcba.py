import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import os
import gc
from pathlib import Path
import json
import warnings

class MemoryEfficientTfIdfPCAAnalyzer:
    def __init__(self, n_components: int = 2, max_features: int = 1000, 
                 cache_dir: str = "./pca_cache", batch_size_days: int = 30):
        """
        Initialize the memory-efficient TF-IDF PCA analyzer.
        
        Args:
            n_components: Number of PCA components to keep
            max_features: Maximum number of features for TF-IDF vectorizer
            cache_dir: Directory to store intermediate results
            batch_size_days: Number of days to process in each batch
        """
        self.n_components = n_components
        self.max_features = max_features
        self.cache_dir = Path(cache_dir)
        self.batch_size_days = batch_size_days
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "batches").mkdir(exist_ok=True)
        (self.cache_dir / "aggregated_texts").mkdir(exist_ok=True)
        
    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare the dataframe by ensuring proper date format and sorting."""
        return (df
                .with_columns([
                    pl.col("date").str.to_date() if df["date"].dtype == pl.Utf8 else pl.col("date")
                ])
                .sort(["date", "qid", "document_id", "sentence_id"])
               )
    
    def get_cache_filename(self, prefix: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache filename for date range."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{prefix}_{start_str}_{end_str}.pkl"
    
    def precompute_aggregated_texts(self, df: pl.DataFrame, force_recompute: bool = False):
        """
        Precompute and cache aggregated texts by qid and date.
        This reduces memory usage by avoiding repeated text concatenation.
        """
        cache_file = self.cache_dir / "aggregated_texts" / "qid_date_texts.pkl"
        
        if cache_file.exists() and not force_recompute:
            print("Loading precomputed aggregated texts from cache...")
            return
        
        print("Precomputing aggregated texts by qid and date...")
        
        # Aggregate texts by qid and date
        aggregated = (df
                     .group_by(["qid", "date"])
                     .agg(pl.col("sentence").str.concat(" "))
                     .sort(["date", "qid"])
                     )
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(aggregated.to_pandas(), f)
        
        print(f"Cached aggregated texts: {aggregated.shape[0]} qid-date combinations")
    
    def load_aggregated_texts(self) -> pd.DataFrame:
        """Load precomputed aggregated texts."""
        cache_file = self.cache_dir / "aggregated_texts" / "qid_date_texts.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def get_texts_for_date_range(self, start_date: datetime, end_date: datetime) -> Dict[str, str]:
        """
        Get aggregated texts for qids within a date range using cached data.
        """
        aggregated_df = self.load_aggregated_texts()
        
        # Filter by date range
        #mask = (aggregated_df['date'] >= start_date) & (aggregated_df['date'] <= end_date)
        #filtered_df = aggregated_df[mask]
        # Convert to pandas Timestamp
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        mask = (aggregated_df['date'] >= start_ts) & (aggregated_df['date'] <= end_ts)
        filtered_df = aggregated_df[mask]
        
        if filtered_df.empty:
            return {}
        
        # Group by qid and concatenate texts across dates
        qid_texts = filtered_df.groupby('qid')['sentence'].apply(lambda x: ' '.join(x)).to_dict()
        
        return qid_texts
    
    def process_single_date(self, target_date: datetime, min_date: datetime, 
                          lookback_months: int = 6) -> Dict:
        """Process a single date and return results."""
        # Calculate window
        preferred_start = target_date - timedelta(days=lookback_months * 30)
        actual_start = max(preferred_start, min_date)
        window_days = (target_date - actual_start).days + 1
        
        # Get texts for this window
        texts = self.get_texts_for_date_range(actual_start, target_date)
        
        if len(texts) < 2:
            return {
                'pca_components': np.array([]),
                'qids': [],
                'explained_variance_ratio': None,
                'n_features': 0,
                'n_qids': 0,
                'window_days': window_days,
                'status': 'insufficient_data'
            }
        
        # Compute TF-IDF and PCA
        pca_result, vectorizer, pca, qids = self.compute_tfidf_pca(texts)
        
        if pca_result.size == 0:
            return {
                'pca_components': np.array([]),
                'qids': qids,
                'explained_variance_ratio': None,
                'n_features': 0,
                'n_qids': len(qids),
                'window_days': window_days,
                'status': 'pca_failed'
            }
        
        return {
            'pca_components': pca_result,
            'qids': qids,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_features': len(vectorizer.get_feature_names_out()),
            'n_qids': len(qids),
            'window_days': window_days,
            'status': 'success'
        }
    
    def compute_tfidf_pca(self, texts: Dict[str, str]) -> Tuple[np.ndarray, TfidfVectorizer, PCA, List[str]]:
        """Compute TF-IDF matrix and apply PCA with memory optimization."""
        if not texts:
            return np.array([]), None, None, []
        
        qids = list(texts.keys())
        documents = list(texts.values())
        
        # Compute TF-IDF with memory efficiency
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            dtype=np.float32  # Use float32 to save memory
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Convert to dense array in smaller chunks if needed
            if tfidf_matrix.shape[0] > 1000:  # Large matrix
                warnings.warn(f"Large matrix ({tfidf_matrix.shape}). Consider increasing batch_size_days.")
            
            # Apply PCA
            n_components = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
            pca = PCA(n_components=n_components)
            
            # Convert to dense for PCA (memory intensive step)
            dense_matrix = tfidf_matrix.toarray().astype(np.float32)
            pca_result = pca.fit_transform(dense_matrix)
            
            # Clean up large objects
            del dense_matrix, tfidf_matrix
            gc.collect()
            
            return pca_result, vectorizer, pca, qids
            
        except Exception as e:
            print(f"Error in TF-IDF/PCA computation: {e}")
            return np.array([]), None, None, qids
    
    def process_date_batch(self, start_date: datetime, end_date: datetime, 
                          min_date: datetime, lookback_months: int = 6) -> Dict[datetime, Dict]:
        """Process a batch of dates and return results."""
        batch_cache_file = self.cache_dir / "batches" / self.get_cache_filename("batch", start_date, end_date)
        print(f"Batch file name: {batch_cache_file}")
        
        # Check if batch already computed
        #if batch_cache_file.exists():
        if os.path.exists(batch_cache_file):
            print(f"Loading cached batch: {start_date} to {end_date}")
            with open(batch_cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Processing batch: {start_date} to {end_date}, check {type(start_date)}, and {type(end_date)}")
        
        # Generate dates for this batch
        current_date = start_date
        batch_results = {}
        
        while current_date <= end_date:
            result = self.process_single_date(current_date, min_date, lookback_months)
            batch_results[current_date] = result
            current_date += timedelta(days=1)
        print('-' * 80)
        print(batch_results)
        print('-' * 80)
        # Cache batch results
        with open(batch_cache_file, 'wb') as f:
            pickle.dump(batch_results, f)
        
        print(f"Cached batch with {len(batch_results)} dates")
        return batch_results
    
    def analyze_all_dates_batched(self, df: pl.DataFrame, lookback_months: int = 6, 
                                 force_recompute: bool = False) -> Dict[datetime, Dict]:
        """
        Perform memory-efficient PCA analysis using batching strategy.
        
        Args:
            df: Polars dataframe
            lookback_months: Number of months to look back
            force_recompute: Whether to force recomputation of cached results
            
        Returns:
            Dictionary mapping dates to analysis results
        """
        df = self.prepare_data(df)
        min_date = df["date"].min()
        max_date = df["date"].max()
        
        print(f"Processing date range: {min_date} to {max_date}")
        print(f"Total days: {(max_date - min_date).days + 1}")
        print(f"Batch size: {self.batch_size_days} days")
        
        # Step 1: Precompute aggregated texts
        self.precompute_aggregated_texts(df, force_recompute)
        
        # Step 2: Process in batches
        current_start = min_date
        all_results = {}
        batch_count = 0
        
        while current_start <= max_date:
            batch_count += 1
            current_end = min(current_start + timedelta(days=self.batch_size_days - 1), max_date)
            
            print(f"\nBatch {batch_count}: {current_start} to {current_end}")
            
            # Process batch
            batch_results = self.process_date_batch(current_start, current_end, min_date, lookback_months)
            all_results.update(batch_results)
            
            # Clean up memory
            gc.collect()
            
            current_start = current_end + timedelta(days=1)
        
        # Print summary
        successful = sum(1 for r in all_results.values() if r['status'] == 'success')
        insufficient = sum(1 for r in all_results.values() if r['status'] == 'insufficient_data')
        failed = sum(1 for r in all_results.values() if r['status'] == 'pca_failed')
        
        print(f"\nFinal Summary:")
        print(f"  Total dates processed: {len(all_results)}")
        print(f"  Successful: {successful}")
        print(f"  Insufficient data: {insufficient}")
        print(f"  PCA failed: {failed}")
        print(f"  Cache directory: {self.cache_dir}")
        
        return all_results
    
    def get_pca_dataframe_streamed(self, results: Dict[datetime, Dict], 
                                  chunk_size: int = 1000) -> pl.DataFrame:
        """
        Convert PCA results to dataframe using streaming to manage memory.
        """
        all_chunks = []
        rows = []
        
        for date, result in results.items():
            pca_components = result['pca_components']
            qids = result['qids']
            status = result['status']
            
            if status != 'success' or pca_components.size == 0:
                rows.append({
                    'date': date,
                    'qid': None,
                    'status': status,
                    'n_qids': result['n_qids']
                })
            else:
                for i, qid in enumerate(qids):
                    row = {
                        'date': date,
                        'qid': qid,
                        'status': status,
                        'n_qids': result['n_qids']
                    }
                    
                    # Add PCA components
                    for j in range(pca_components.shape[1]):
                        row[f'pc_{j+1}'] = pca_components[i, j]
                    
                    rows.append(row)
            
            # Process in chunks to manage memory
            if len(rows) >= chunk_size:
                chunk_df = pl.DataFrame(rows)
                all_chunks.append(chunk_df)
                rows = []
                gc.collect()
        
        # Process remaining rows
        if rows:
            chunk_df = pl.DataFrame(rows)
            all_chunks.append(chunk_df)
        
        # Concatenate all chunks
        if all_chunks:
            return pl.concat(all_chunks)
        else:
            return pl.DataFrame()
    
    def clear_cache(self):
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            (self.cache_dir / "batches").mkdir(exist_ok=True)
            (self.cache_dir / "aggregated_texts").mkdir(exist_ok=True)
        print("Cache cleared")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        cache_info = {
            'cache_dir': str(self.cache_dir),
            'aggregated_texts_exists': (self.cache_dir / "aggregated_texts" / "qid_date_texts.pkl").exists(),
            'batch_files': len(list((self.cache_dir / "batches").glob("*.pkl"))),
            'total_cache_size_mb': 0
        }
        
        # Calculate cache size
        for file_path in self.cache_dir.rglob("*.pkl"):
            cache_info['total_cache_size_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        cache_info['total_cache_size_mb'] = round(cache_info['total_cache_size_mb'], 2)
        
        return cache_info
    

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
                    'date': filing_date,
                    'sentence': text
                })
        
        # Progress indicator
        if (qid_idx + 1) % 50 == 0:
            print(f"Generated data for {qid_idx + 1}/{n_qids} qids...")
    
    print(f"Dataset generation complete. Total records: {len(data_records):,}")
    syn_data = pl.DataFrame(data_records)
    syn_data = syn_data.with_columns(
        pl.col("date").str.to_date().alias("date")
    )
    print(f"Data samples:\n {syn_data.head()}")
    return syn_data


# Example usage for large datasets
def main_large_dataset():
    """Example usage for large 10-year dataset."""
    
    # Initialize with appropriate settings for large data
    analyzer = MemoryEfficientTfIdfPCAAnalyzer(
        n_components=2,
        max_features=500,  # Reduced for memory efficiency
        cache_dir="./pca_cache_10years",
        batch_size_days=15  # Smaller batches for large datasets
    )
    
    # Load your large dataset
    # df = pl.read_csv("large_dataset.csv")
    # or read in chunks if too large for memory:
    # df = pl.scan_csv("large_dataset.csv").collect()
    df = generate_large_test_dataset(n_qids=50, n_docs_per_qid=10, n_sentences_per_doc=8)
    
    # Run batched analysis
    results = analyzer.analyze_all_dates_batched(df, lookback_months=6)
    
    # Get results as dataframe (streamed for memory efficiency)
    pca_df = analyzer.get_pca_dataframe_streamed(results, chunk_size=500)
    
    # Check cache info
    cache_info = analyzer.get_cache_info()
    print("Cache info:", cache_info)
    
    # Save final results (optional)
    # pca_df.write_parquet("pca_results_10years.parquet")
    
    # Memory usage tips:
    # 1. Process in smaller batches if you hit memory limits
    # 2. Reduce max_features if vocabulary is too large
    # 3. Use fewer PCA components if needed
    # 4. Clear cache periodically if disk space is limited
    

if __name__ == "__main__":
    main_large_dataset()
