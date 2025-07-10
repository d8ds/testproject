import polars as pl

# Example: small sample table
df = pl.DataFrame({
    "qid": ["AAPL", "AAPL", "AAPL", "MSFT"],
    "document_id": ["doc1", "doc1", "doc2", "doc3"],
    "sentence_id": [1, 2, 1, 1],
    "sentence_text": [
        "The company reported strong earnings.",
        "Revenue growth exceeded expectations.",
        "New product launch next quarter.",
        "Microsoft announced a dividend increase."
    ],
    "filing_date": ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-03"]
})---# Concatenate all sentences per document
docs_df = (
    df.groupby(["qid", "document_id", "filing_date"])
    .agg(
        pl.col("sentence_text").str.concat(" ")
    )
    .rename({"sentence_text": "document_text"})
)

print(docs_df)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Get texts
texts = docs_df["document_text"].to_list()

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(texts).toarray()

# PCA
pca = PCA(n_components=1)  # for simplicity, 1 component
X_pca = pca.fit_transform(X)
import pandas as pd

# Combine with Polars
docs_df = docs_df.with_columns([
    pl.Series("pc1", X_pca.flatten())
])

print(docs_df)
# If you have multiple docs per (qid, filing_date), average
factor_df = (
    docs_df.groupby(["qid", "filing_date"])
    .agg(
        pl.col("pc1").mean().alias("alpha_factor")
    )
)

print(factor_df)

def process_8k_filings_lsa(df, n_components=50, max_features=5000):
    """Complete pipeline for 8K filing alpha generation"""
    
    # 1. Aggregate to document level
    doc_df = df.group_by(['qid', 'document_id', 'filing_date']).agg(
        pl.col('sentence_text').str.concat(' ').alias('full_text')
    )
    
    # 2. Create and fit LSA pipeline
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents='ascii'
    )
    
    # 3. Vectorize text
    X_tfidf = vectorizer.fit_transform(doc_df['full_text'])
    
    # 4. Apply TruncatedSVD
    svd = TruncatedSVD(
        n_components=n_components,
        algorithm='randomized',
        random_state=42
    )
    X_lsa = svd.fit_transform(X_tfidf)
    
    # 5. Add LSA features to dataframe
    feature_cols = [f'lsa_{i}' for i in range(n_components)]
    lsa_df = pl.DataFrame(X_lsa, schema=feature_cols)
    doc_df = pl.concat([doc_df, lsa_df], how='horizontal')
    
    # 6. Aggregate to stock-date level
    stock_df = doc_df.group_by(['qid', 'filing_date']).agg([
        pl.col(f'lsa_{i}').mean().alias(f'lsa_{i}_mean') 
        for i in range(n_components)
    ] + [
        pl.col(f'lsa_{i}').std().alias(f'lsa_{i}_std') 
        for i in range(n_components)
    ])
    
    return stock_df, svd, vectorizer

# Usage
final_df, svd_model, tfidf_model = process_8k_filings_lsa(
    your_df, 
    n_components=50, 
    max_features=5000
)
