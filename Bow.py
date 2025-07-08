# 将同一document_id的所有句子合并
def aggregate_sentences_by_document(df):
    doc_aggregated = df.groupby(['qid', 'document_id', 'filing_date']).agg({
        'sentence_text': ' '.join
    }).reset_index()
    return doc_aggregated

# 或者保持句子级别，但添加文档权重
def weight_sentences_by_document(df):
    df['doc_sentence_count'] = df.groupby('document_id')['sentence_id'].transform('count')
    df['sentence_weight'] = 1.0 / df['doc_sentence_count']
    return df


def create_time_windows(df, window_months=3):
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df = df.sort_values('filing_date')
    
    # 创建滚动窗口
    windows = []
    end_date = df['filing_date'].max()
    
    for i in range(window_months, len(df['filing_date'].dt.to_period('M').unique()) + 1):
        window_end = end_date - pd.DateOffset(months=i-window_months)
        window_start = end_date - pd.DateOffset(months=i)
        
        window_data = df[(df['filing_date'] >= window_start) & 
                        (df['filing_date'] < window_end)]
        windows.append((window_start, window_end, window_data))
    
    return windows

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

def advanced_text_preprocessing(text):
    # 金融专业词汇保护
    financial_terms = ['EBITDA', 'GAAP', 'non-GAAP', 'cash flow', 'revenue', 'margin']
    protected_terms = {}
    
    for i, term in enumerate(financial_terms):
        placeholder = f"PROTECTED_TERM_{i}"
        protected_terms[placeholder] = term
        text = text.replace(term, placeholder)
    
    # 标准预处理
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    
    # 还原保护词汇
    for placeholder, term in protected_terms.items():
        text = text.replace(placeholder, term)
    
    return text

# 多层次特征提取
def extract_textual_features(df, n_components=50):
    # 1. TF-IDF特征
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=0.01,
        max_df=0.95,
        preprocessor=advanced_text_preprocessing,
        stop_words='english'
    )
    
    # 2. 情感特征
    def get_sentiment_features(text):
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    # 3. 可读性特征
    def get_readability_features(text):
        sentences = text.split('.')
        words = text.split()
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'total_words': len(words),
            'unique_words': len(set(words))
        }
    
    return tfidf_vectorizer, get_sentiment_features, get_readability_features


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class FilingFactorAnalyzer:
    def __init__(self, n_components=20, window_months=6):
        self.n_components = n_components
        self.window_months = window_months
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = None
        
    def fit_transform_window(self, window_data):
        # 1. 文本向量化
        texts = window_data['sentence_text'].values
        
        # TF-IDF特征
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=0.02,
                max_df=0.8
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # 2. 公司级别聚合
        company_features = self._aggregate_by_company(
            window_data, tfidf_matrix.toarray()
        )
        
        # 3. 标准化和PCA
        scaled_features = self.scaler.fit_transform(company_features)
        pca_features = self.pca.fit_transform(scaled_features)
        
        return pca_features, company_features
    
    def _aggregate_by_company(self, window_data, tfidf_matrix):
        # 按公司聚合特征
        company_groups = window_data.groupby('qid')
        aggregated_features = []
        
        for company_id, group in company_groups:
            # 获取该公司在当前窗口的所有文档特征
            company_indices = group.index
            company_tfidf = tfidf_matrix[company_indices]
            
            # 聚合方法：均值、最大值、标准差
            company_mean = np.mean(company_tfidf, axis=0)
            company_max = np.max(company_tfidf, axis=0)
            company_std = np.std(company_tfidf, axis=0)
            
            # 组合特征
            combined_features = np.concatenate([
                company_mean, company_max, company_std
            ])
            
            aggregated_features.append(combined_features)
        
        return np.array(aggregated_features)
    
    def get_factor_interpretation(self, feature_names):
        # 解释主成分
        components = self.pca.components_
        factor_interpretation = {}
        
        for i, component in enumerate(components):
            # 找到贡献最大的特征
            top_indices = np.argsort(np.abs(component))[-10:]
            top_features = [(feature_names[idx], component[idx]) 
                           for idx in top_indices]
            
            factor_interpretation[f'Factor_{i+1}'] = {
                'explained_variance': self.pca.explained_variance_ratio_[i],
                'top_features': top_features
            }
        
        return factor_interpretation


def rolling_factor_extraction(df, analyzer, window_months=6, step_months=1):
    """
    滚动窗口提取因子
    """
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df = df.sort_values('filing_date')
    
    results = []
    end_date = df['filing_date'].max()
    start_date = df['filing_date'].min()
    
    current_date = start_date + pd.DateOffset(months=window_months)
    
    while current_date <= end_date:
        # 定义时间窗口
        window_start = current_date - pd.DateOffset(months=window_months)
        window_end = current_date
        
        # 提取窗口数据
        window_data = df[(df['filing_date'] >= window_start) & 
                        (df['filing_date'] < window_end)]
        
        if len(window_data) > 0:
            # 提取因子
            factors, raw_features = analyzer.fit_transform_window(window_data)
            
            # 保存结果
            companies = window_data['qid'].unique()
            for i, company in enumerate(companies):
                results.append({
                    'qid': company,
                    'window_end': window_end,
                    'factors': factors[i] if i < len(factors) else None
                })
        
        current_date += pd.DateOffset(months=step_months)
    
    return pd.DataFrame(results)

def generate_factor_signals(factor_df, lookback_periods=3):
    """
    基于因子生成交易信号
    """
    signals = []
    
    for company in factor_df['qid'].unique():
        company_data = factor_df[factor_df['qid'] == company].sort_values('window_end')
        
        if len(company_data) < lookback_periods:
            continue
            
        for i in range(lookback_periods, len(company_data)):
            current_factors = company_data.iloc[i]['factors']
            historical_factors = [company_data.iloc[j]['factors'] 
                                for j in range(i-lookback_periods, i)]
            
            # 计算因子变化
            factor_momentum = calculate_factor_momentum(current_factors, historical_factors)
            factor_mean_reversion = calculate_factor_mean_reversion(current_factors, historical_factors)
            
            signals.append({
                'qid': company,
                'date': company_data.iloc[i]['window_end'],
                'momentum_signal': factor_momentum,
                'mean_reversion_signal': factor_mean_reversion
            })
    
    return pd.DataFrame(signals)

def calculate_factor_momentum(current, historical):
    # 计算因子动量
    historical_mean = np.mean(historical, axis=0)
    momentum = current - historical_mean
    return np.linalg.norm(momentum)

def calculate_factor_mean_reversion(current, historical):
    # 计算因子均值回归
    historical_std = np.std(historical, axis=0)
    z_score = (current - np.mean(historical, axis=0)) / (historical_std + 1e-8)
    return -np.mean(z_score)  # 负号表示均值回归

# 初始化
analyzer = FilingFactorAnalyzer(n_components=15, window_months=6)

# 提取滚动因子
factor_results = rolling_factor_extraction(
    df, analyzer, window_months=6, step_months=1
)

# 生成交易信号
trading_signals = generate_factor_signals(factor_results)

# 因子解释
feature_names = analyzer.tfidf_vectorizer.get_feature_names_out()
factor_interpretation = analyzer.get_factor_interpretation(feature_names)

# 打印因子解释
for factor_name, interpretation in factor_interpretation.items():
    print(f"\n{factor_name} (解释方差: {interpretation['explained_variance']:.3f})")
    print("主要特征:")
    for feature, weight in interpretation['top_features']:
        print(f"  {feature}: {weight:.3f}")

'''
7. 性能优化建议

并行处理：使用multiprocessing处理不同时间窗口
特征缓存：缓存TF-IDF矩阵避免重复计算
增量更新：只处理新的filing数据
维度优化：使用Truncated SVD替代完整PCA
内存管理：使用sparse矩阵处理大规模文本数据

这个策略可以有效地从8K filing中提取有价值的量化信号，关键是要根据实际数据特征调整参数，并结合domain knowledge优化特征工程。
'''
