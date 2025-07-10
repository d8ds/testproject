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
