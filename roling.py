import polars as pl
import numpy as np
from collections import Counter
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FilingEntropySignal:
    def __init__(self, df: pl.DataFrame, n_months: int = 3):
        """
        Initialize the entropy signal extractor.
        
        Args:
            df: Polars DataFrame with columns [qid, document_id, sentence_id, sentence, date]
            n_months: Number of months to look back for signal calculation
        """
        self.df = df
        self.n_months = n_months
        self.vocab_cache = {}
        
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text efficiently."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        return words
    
    def calculate_entropy(self, tokens: List[str]) -> float:
        """Calculate Shannon entropy of token distribution."""
        if not tokens:
            return 0.0
            
        # Count token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate entropy
        entropy = 0.0
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * np.log2(prob)
            
        return entropy
    
    def get_document_entropy(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate various entropy measures for a document."""
        # Combine all sentences in the document
        all_text = " ".join(sentences)
        tokens = self.preprocess_text(all_text)
        
        if not tokens:
            return {
                'word_entropy': 0.0,
                'sentence_entropy': 0.0,
                'avg_sentence_length': 0.0,
                'unique_word_ratio': 0.0,
                'total_words': 0
            }
        
        # Word-level entropy
        word_entropy = self.calculate_entropy(tokens)
        
        # Sentence-level entropy (based on sentence lengths)
        sentence_lengths = [len(self.preprocess_text(sent)) for sent in sentences]
        sentence_entropy = self.calculate_entropy([str(length) for length in sentence_lengths])
        
        # Additional features
        unique_words = len(set(tokens))
        unique_word_ratio = unique_words / len(tokens) if tokens else 0
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        return {
            'word_entropy': word_entropy,
            'sentence_entropy': sentence_entropy,
            'avg_sentence_length': avg_sentence_length,
            'unique_word_ratio': unique_word_ratio,
            'total_words': len(tokens)
        }
    
    def calculate_filing_entropy(self, reference_date: Optional[str] = None) -> pl.DataFrame:
        """
        Calculate entropy measures for each filing within the specified time window.
        
        Args:
            reference_date: Reference date to look back from (default: latest date in data)
        """
        # Convert date column to datetime if it's not already
        df_with_date = self.df.with_columns([
            pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False)
        ])
        
        # Determine reference date
        if reference_date is None:
            ref_date = df_with_date.select(pl.col('date').max()).item()
        else:
            ref_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
        
        # Filter data for the past n months
        cutoff_date = ref_date - timedelta(days=self.n_months * 30)
        
        filtered_df = df_with_date.filter(
            pl.col('date') >= cutoff_date
        )
        
        # Group by qid and document_id to calculate entropy per filing
        result_list = []
        
        # Process each document
        for qid_doc in filtered_df.select(['qid', 'document_id', 'date']).unique().iter_rows():
            qid, doc_id, doc_date = qid_doc
            
            # Get all sentences for this document
            doc_sentences = filtered_df.filter(
                (pl.col('qid') == qid) & (pl.col('document_id') == doc_id)
            ).select('sentence').to_series().to_list()
            
            # Calculate entropy measures
            entropy_measures = self.get_document_entropy(doc_sentences)
            
            # Add metadata
            entropy_measures.update({
                'qid': qid,
                'document_id': doc_id,
                'date': doc_date,
                'num_sentences': len(doc_sentences)
            })
            
            result_list.append(entropy_measures)
        
        # Convert to DataFrame
        entropy_df = pl.DataFrame(result_list)
        
        return entropy_df
    
    def calculate_relative_entropy_signal(self, entropy_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate relative entropy signals by comparing each filing to historical average.
        """
        # Calculate historical averages per qid
        historical_stats = entropy_df.group_by('qid').agg([
            pl.col('word_entropy').mean().alias('avg_word_entropy'),
            pl.col('word_entropy').std().alias('std_word_entropy'),
            pl.col('sentence_entropy').mean().alias('avg_sentence_entropy'),
            pl.col('sentence_entropy').std().alias('std_sentence_entropy'),
            pl.col('unique_word_ratio').mean().alias('avg_unique_word_ratio'),
            pl.col('unique_word_ratio').std().alias('std_unique_word_ratio'),
            pl.col('total_words').mean().alias('avg_total_words'),
            pl.col('total_words').std().alias('std_total_words')
        ])
        
        # Join with entropy data
        signal_df = entropy_df.join(historical_stats, on='qid')
        
        # Calculate z-scores (standardized signals)
        signal_df = signal_df.with_columns([
            ((pl.col('word_entropy') - pl.col('avg_word_entropy')) / 
             pl.col('std_word_entropy')).alias('word_entropy_zscore'),
            ((pl.col('sentence_entropy') - pl.col('avg_sentence_entropy')) / 
             pl.col('std_sentence_entropy')).alias('sentence_entropy_zscore'),
            ((pl.col('unique_word_ratio') - pl.col('avg_unique_word_ratio')) / 
             pl.col('std_unique_word_ratio')).alias('unique_ratio_zscore'),
            ((pl.col('total_words') - pl.col('avg_total_words')) / 
             pl.col('std_total_words')).alias('word_count_zscore')
        ])
        
        # Create composite signal
        signal_df = signal_df.with_columns([
            (pl.col('word_entropy_zscore') * 0.4 + 
             pl.col('sentence_entropy_zscore') * 0.3 + 
             pl.col('unique_ratio_zscore') * 0.2 + 
             pl.col('word_count_zscore') * 0.1).alias('composite_entropy_signal')
        ])
        
        return signal_df
    
    def get_anomaly_filings(self, signal_df: pl.DataFrame, 
                          threshold: float = 2.0) -> pl.DataFrame:
        """
        Identify filings with anomalous entropy patterns.
        
        Args:
            signal_df: DataFrame with entropy signals
            threshold: Z-score threshold for anomaly detection
        """
        anomalies = signal_df.filter(
            pl.col('composite_entropy_signal').abs() > threshold
        ).sort('composite_entropy_signal', descending=True)
        
        return anomalies
    
    def run_analysis(self, reference_date: Optional[str] = None, 
                    anomaly_threshold: float = 2.0) -> Dict[str, pl.DataFrame]:
        """
        Run the complete entropy analysis pipeline.
        
        Returns:
            Dictionary containing entropy_df, signal_df, and anomalies
        """
        print(f"Calculating entropy for filings in the past {self.n_months} months...")
        
        # Calculate basic entropy measures
        entropy_df = self.calculate_filing_entropy(reference_date)
        print(f"Processed {len(entropy_df)} filings")
        
        # Calculate relative signals
        signal_df = self.calculate_relative_entropy_signal(entropy_df)
        print("Calculated relative entropy signals")
        
        # Find anomalies
        anomalies = self.get_anomaly_filings(signal_df, anomaly_threshold)
        print(f"Found {len(anomalies)} anomalous filings")
        
        return {
            'entropy_df': entropy_df,
            'signal_df': signal_df,
            'anomalies': anomalies
        }

# Example usage and performance optimization
def optimize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Optimize DataFrame for memory and speed."""
    return df.with_columns([
        pl.col('qid').cast(pl.Categorical),
        pl.col('document_id').cast(pl.Categorical),
        pl.col('sentence_id').cast(pl.UInt32),
        pl.col('date').cast(pl.Date)
    ])

# Example usage
if __name__ == "__main__":
    # Sample data creation (replace with your actual data)
    sample_data = {
        'qid': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT'] * 100,
        'document_id': ['doc1', 'doc1', 'doc2', 'doc2', 'doc3'] * 100,
        'sentence_id': list(range(500)),
        'sentence': ['This is a sample sentence about financial performance.'] * 500,
        'date': ['2024-01-15', '2024-01-15', '2024-02-20', '2024-02-20', '2024-03-10'] * 100
    }
    
    df = pl.DataFrame(sample_data)
    df = optimize_dataframe(df)
    
    # Initialize analyzer
    analyzer = FilingEntropySignal(df, n_months=3)
    
    # Run analysis
    results = analyzer.run_analysis(anomaly_threshold=1.5)
    
    # Display results
    print("\nTop 5 Entropy Signals:")
    print(results['signal_df'].select([
        'qid', 'document_id', 'date', 'word_entropy', 
        'composite_entropy_signal'
    ]).sort('composite_entropy_signal', descending=True).head(5))
    
    print("\nAnomalous Filings:")
    print(results['anomalies'].select([
        'qid', 'document_id', 'date', 'composite_entropy_signal'
    ]).head(10))

# Initialize with your data
analyzer = FilingEntropySignal(df, n_months=3)

# Run complete analysis
results = analyzer.run_analysis(anomaly_threshold=1.5)

# Access results
entropy_df = results['entropy_df']      # Basic entropy measures
signal_df = results['signal_df']        # Relative signals with z-scores
anomalies = results['anomalies']        # Unusual filings
