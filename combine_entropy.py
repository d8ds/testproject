import polars as pl
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import jensenshannon
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EntropyFactorEngine:
    """
    基于信息熵的8K Filing因子挖掘引擎
    """
    
    def __init__(self, window_days: int = 90):
        """
        初始化
        
        Args:
            window_days: 计算熵的时间窗口（天）
        """
        self.window_days = window_days
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text or pd.isna(text):
            return ""
        
        # 转小写，去除特殊字符
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        # 去除多余空格
        text = ' '.join(text.split())
        return text
    
    def calculate_word_entropy(self, text_list: List[str]) -> float:
        """
        计算词汇分布熵
        
        基于Shannon熵公式: H(X) = -Σ p(x) * log2(p(x))
        """
        if not text_list:
            return 0.0
            
        # 合并所有文本
        combined_text = ' '.join([self.preprocess_text(t) for t in text_list])
        
        if not combined_text:
            return 0.0
            
        # 计算词频
        words = combined_text.split()
        word_counts = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # 计算概率分布
        probabilities = [count / total_words for count in word_counts.values()]
        
        # 计算熵
        return entropy(probabilities, base=2)
    
    def calculate_sentence_length_entropy(self, text_list: List[str]) -> float:
        """
        计算句子长度分布熵
        """
        if not text_list:
            return 0.0
            
        lengths = [len(text.split()) for text in text_list if text]
        
        if not lengths:
            return 0.0
            
        # 将长度分桶
        max_len = max(lengths)
        bins = min(10, max_len)  # 最多10个桶
        
        if bins <= 1:
            return 0.0
            
        hist, _ = np.histogram(lengths, bins=bins)
        hist = hist[hist > 0]  # 去除空桶
        
        if len(hist) <= 1:
            return 0.0
            
        # 计算概率分布
        probabilities = hist / np.sum(hist)
        
        return entropy(probabilities, base=2)
    
    def calculate_tfidf_entropy(self, current_texts: List[str], 
                              historical_texts: List[str]) -> float:
        """
        基于TF-IDF的语义熵
        衡量当前文档与历史文档的语义差异
        """
        if not current_texts or not historical_texts:
            return 0.0
            
        try:
            # 预处理文本
            current_clean = [self.preprocess_text(t) for t in current_texts]
            historical_clean = [self.preprocess_text(t) for t in historical_texts]
            
            # 过滤空文本
            current_clean = [t for t in current_clean if t]
            historical_clean = [t for t in historical_clean if t]
            
            if not current_clean or not historical_clean:
                return 0.0
            
            # 合并文本
            current_doc = ' '.join(current_clean)
            historical_doc = ' '.join(historical_clean)
            
            # 计算TF-IDF
            docs = [current_doc, historical_doc]
            tfidf_matrix = self.vectorizer.fit_transform(docs)
            
            if tfidf_matrix.shape[0] < 2:
                return 0.0
            
            # 计算JS散度作为语义熵的度量
            vec1 = tfidf_matrix[0].toarray().flatten()
            vec2 = tfidf_matrix[1].toarray().flatten()
            
            # 归一化为概率分布
            vec1 = vec1 / (np.sum(vec1) + 1e-10)
            vec2 = vec2 / (np.sum(vec2) + 1e-10)
            
            # 计算JS散度
            js_div = jensenshannon(vec1, vec2)
            
            return js_div
            
        except Exception as e:
            print(f"TF-IDF熵计算错误: {e}")
            return 0.0
    
    def calculate_temporal_entropy_change(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算时间序列熵变化
        """
        results = []
        
        # 按股票分组
        for qid in df['qid'].unique():
            qid_data = df.filter(pl.col('qid') == qid).sort('filing_date')
            
            # 获取所有唯一的filing日期
            filing_dates = qid_data['filing_date'].unique().sort()
            
            for current_date in filing_dates:
                # 获取当前日期的数据
                current_data = qid_data.filter(pl.col('filing_date') == current_date)
                current_texts = current_data['sentence_text'].to_list()
                
                # 获取历史数据（过去window_days天）
                start_date = current_date - pl.duration(days=self.window_days)
                historical_data = qid_data.filter(
                    (pl.col('filing_date') >= start_date) & 
                    (pl.col('filing_date') < current_date)
                )
                historical_texts = historical_data['sentence_text'].to_list()
                
                # 计算各种熵指标
                word_entropy = self.calculate_word_entropy(current_texts)
                length_entropy = self.calculate_sentence_length_entropy(current_texts)
                
                # 计算与历史的语义熵差异
                semantic_entropy = 0.0
                if historical_texts:
                    semantic_entropy = self.calculate_tfidf_entropy(current_texts, historical_texts)
                
                # 计算历史熵基线
                historical_word_entropy = self.calculate_word_entropy(historical_texts)
                historical_length_entropy = self.calculate_sentence_length_entropy(historical_texts)
                
                # 计算熵变化率
                word_entropy_change = (word_entropy - historical_word_entropy) / (historical_word_entropy + 1e-10)
                length_entropy_change = (length_entropy - historical_length_entropy) / (historical_length_entropy + 1e-10)
                
                # 计算filing频率熵（信息披露频率的不确定性）
                filing_freq_entropy = 0.0
                if len(historical_data) > 0:
                    doc_counts = historical_data.group_by('document_id').len()
                    freq_dist = doc_counts['len'].to_list()
                    if len(freq_dist) > 1:
                        freq_probs = np.array(freq_dist) / np.sum(freq_dist)
                        filing_freq_entropy = entropy(freq_probs, base=2)
                
                results.append({
                    'qid': qid,
                    'filing_date': current_date,
                    'word_entropy': word_entropy,
                    'length_entropy': length_entropy,
                    'semantic_entropy': semantic_entropy,
                    'word_entropy_change': word_entropy_change,
                    'length_entropy_change': length_entropy_change,
                    'filing_freq_entropy': filing_freq_entropy,
                    'historical_word_entropy': historical_word_entropy,
                    'historical_length_entropy': historical_length_entropy,
                    'current_filing_count': len(current_data),
                    'historical_filing_count': len(historical_data)
                })
        
        return pl.DataFrame(results)
    
    def create_composite_entropy_factors(self, entropy_df: pl.DataFrame) -> pl.DataFrame:
        """
        创建复合熵因子
        """
        return entropy_df.with_columns([
            # 综合信息熵指标
            (pl.col('word_entropy') * 0.4 + 
             pl.col('length_entropy') * 0.3 + 
             pl.col('semantic_entropy') * 0.3).alias('composite_entropy'),
            
            # 信息变化强度
            (pl.col('word_entropy_change').abs() * 0.5 + 
             pl.col('length_entropy_change').abs() * 0.5).alias('info_change_intensity'),
            
            # 信息复杂度（高熵表示信息复杂）
            (pl.col('word_entropy') + pl.col('filing_freq_entropy')).alias('info_complexity'),
            
            # 信息异常度（与历史相比的异常程度）
            (pl.col('semantic_entropy') * 
             (1 + pl.col('word_entropy_change').abs())).alias('info_anomaly'),
            
            # 信息密度（单位filing的信息量）
            (pl.col('word_entropy') / 
             (pl.col('current_filing_count') + 1)).alias('info_density')
        ])
    
    def rank_and_normalize_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        因子排名和标准化
        """
        factor_cols = [
            'word_entropy', 'length_entropy', 'semantic_entropy',
            'word_entropy_change', 'length_entropy_change', 'filing_freq_entropy',
            'composite_entropy', 'info_change_intensity', 'info_complexity',
            'info_anomaly', 'info_density'
        ]
        
        # 按日期分组进行截面标准化
        result_df = df.sort(['filing_date', 'qid'])
        
        for col in factor_cols:
            # 计算分位数排名
            result_df = result_df.with_columns([
                pl.col(col).rank(method='average').over('filing_date').alias(f'{col}_rank'),
                # Z-score标准化
                ((pl.col(col) - pl.col(col).mean().over('filing_date')) / 
                 (pl.col(col).std().over('filing_date') + 1e-10)).alias(f'{col}_zscore')
            ])
        
        return result_df

def demo_usage():
    """演示用法"""
    # 模拟数据
    np.random.seed(42)
    
    # 创建模拟的8K filing数据
    dates = pl.date_range(
        start=pl.date(2023, 1, 1),
        end=pl.date(2024, 1, 1),
        interval='1d'
    )
    
    qids = [f'QID_{i:03d}' for i in range(1, 11)]  # 10只股票
    
    data = []
    for qid in qids:
        # 每只股票随机选择一些日期进行filing
        filing_dates = np.random.choice(dates, size=np.random.randint(20, 50), replace=False)
        
        for filing_date in filing_dates:
            doc_id = f'DOC_{qid}_{filing_date}'
            # 每个文档包含多个句子
            num_sentences = np.random.randint(5, 20)
            
            for sent_id in range(num_sentences):
                # 生成模拟句子（简化版）
                sentence_templates = [
                    f"The company reported {np.random.choice(['strong', 'weak', 'stable'])} performance in {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])}.",
                    f"Revenue {np.random.choice(['increased', 'decreased', 'remained stable'])} by {np.random.randint(1, 30)}% compared to previous period.",
                    f"Management expects {np.random.choice(['growth', 'challenges', 'opportunities'])} in the upcoming quarter.",
                    f"The board approved a {np.random.choice(['dividend', 'share buyback', 'investment'])} program.",
                    f"Regulatory {np.random.choice(['approval', 'review', 'compliance'])} was {np.random.choice(['obtained', 'pending', 'required'])}."
                ]
                
                sentence = np.random.choice(sentence_templates)
                
                data.append({
                    'qid': qid,
                    'document_id': doc_id,
                    'sentence_id': f'{doc_id}_S{sent_id}',
                    'sentence_text': sentence,
                    'filing_date': filing_date
                })
    
    df = pl.DataFrame(data)
    print("原始数据示例:")
    print(df.head())
    print(f"\n数据维度: {df.shape}")
    
    # 创建因子引擎
    engine = EntropyFactorEngine(window_days=90)
    
    # 计算熵因子
    print("\n开始计算熵因子...")
    entropy_factors = engine.calculate_temporal_entropy_change(df)
    
    # 创建复合因子
    print("创建复合因子...")
    composite_factors = engine.create_composite_entropy_factors(entropy_factors)
    
    # 标准化因子
    print("标准化因子...")
    final_factors = engine.rank_and_normalize_factors(composite_factors)
    
    print("\n最终因子数据:")
    print(final_factors.select([
        'qid', 'filing_date', 'composite_entropy', 'info_change_intensity',
        'info_complexity', 'info_anomaly', 'info_density'
    ]).head(10))
    
    print(f"\n最终因子数据维度: {final_factors.shape}")
    
    # 因子有效性初步分析
    print("\n因子统计信息:")
    factor_stats = final_factors.select([
        'composite_entropy', 'info_change_intensity', 'info_complexity',
        'info_anomaly', 'info_density'
    ]).describe()
    print(factor_stats)
    
    return final_factors

if __name__ == "__main__":
    # 运行演示
    factors = demo_usage()
    
    print("\n=== 因子解释 ===")
    print("1. word_entropy: 词汇分布熵，衡量用词的多样性")
    print("2. length_entropy: 句子长度分布熵，衡量句子结构的复杂性")
    print("3. semantic_entropy: 语义熵，衡量与历史内容的语义差异")
    print("4. composite_entropy: 综合熵指标，综合考虑多个维度")
    print("5. info_change_intensity: 信息变化强度，衡量信息披露的变化程度")
    print("6. info_complexity: 信息复杂度，衡量信息的复杂程度")
    print("7. info_anomaly: 信息异常度，衡量信息披露的异常程度")
    print("8. info_density: 信息密度，衡量单位filing的信息量")
    
    print("\n=== 因子应用建议 ===")
    print("1. 高熵值可能表示公司面临重大变化或不确定性")
    print("2. 熵的突然变化可能预示着重要事件或业绩转折")
    print("3. 可以结合股价数据验证因子的预测能力")
    print("4. 建议进行因子正交化，去除共线性")
    print("5. 可以按行业或市值分组，提高因子的有效性")
