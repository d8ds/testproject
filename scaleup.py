import polars as pl
import numpy as np
from collections import Counter
from scipy.stats import entropy
import re
from typing import Dict, List, Tuple, Optional, Iterator
import warnings
import gc
import os
from pathlib import Path
import tempfile
import hashlib
warnings.filterwarnings('ignore')

class ScalableEntropyFactorEngine:
    """
    超大规模数据的熵因子计算引擎
    支持上亿级别数据，使用分块处理和磁盘缓存
    """
    
    def __init__(self, 
                 window_days: int = 90,
                 chunk_size: int = 1000000,  # 每块100万行
                 temp_dir: str = "/tmp/entropy_factors",
                 n_vocab_buckets: int = 10000,  # 词汇哈希桶数量
                 enable_disk_cache: bool = True):
        """
        初始化
        
        Args:
            window_days: 时间窗口天数
            chunk_size: 分块大小
            temp_dir: 临时文件目录
            n_vocab_buckets: 词汇哈希桶数量（控制内存使用）
            enable_disk_cache: 是否启用磁盘缓存
        """
        self.window_days = window_days
        self.chunk_size = chunk_size
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.n_vocab_buckets = n_vocab_buckets
        self.enable_disk_cache = enable_disk_cache
        
        # 清理临时文件
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*.parquet"):
                file.unlink()
    
    def _hash_word(self, word: str) -> int:
        """将词汇哈希到固定数量的桶中"""
        return int(hashlib.md5(word.encode()).hexdigest(), 16) % self.n_vocab_buckets
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.temp_dir / f"{cache_key}.parquet"
    
    def _save_cache(self, df: pl.DataFrame, cache_key: str):
        """保存缓存"""
        if self.enable_disk_cache:
            cache_path = self._get_cache_path(cache_key)
            df.write_parquet(cache_path)
    
    def _load_cache(self, cache_key: str) -> Optional[pl.DataFrame]:
        """加载缓存"""
        if not self.enable_disk_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            return pl.read_parquet(cache_path)
        return None
    
    def preprocess_text_streaming(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        流式文本预处理
        避免全部数据同时加载到内存
        """
        print("开始流式文本预处理...")
        
        # 基本清洗和特征提取
        processed = df.with_columns([
            # 文本清洗
            pl.col('sentence_text')
            .str.to_lowercase()
            .str.replace_all(r'[^a-zA-Z\s]', ' ')
            .str.replace_all(r'\s+', ' ')
            .str.strip()
            .alias('clean_text'),
            
            # 基本统计特征
            pl.col('sentence_text').str.len_chars().alias('char_count'),
            pl.col('sentence_text').str.split(' ').list.len().alias('word_count'),
            
            # 词汇多样性的快速估计（无需完整分词）
            pl.col('sentence_text').str.split(' ').list.unique().list.len().alias('unique_word_count'),
            
            # 句子复杂度代理指标
            pl.col('sentence_text').str.count_matches(r'[,;:]').alias('punctuation_count'),
            pl.col('sentence_text').str.count_matches(r'\b(and|or|but|because|therefore|however)\b').alias('conjunction_count')
        ]).select([
            'qid', 'document_id', 'filing_date', 'clean_text', 
            'char_count', 'word_count', 'unique_word_count', 
            'punctuation_count', 'conjunction_count'
        ])
        
        return processed
    
    def calculate_document_level_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算文档级别特征
        避免句子级别的复杂计算
        """
        print("计算文档级别特征...")
        
        # 文档级别聚合
        doc_features = df.group_by(['qid', 'document_id', 'filing_date']).agg([
            # 基本统计
            pl.len().alias('sentence_count'),
            pl.col('word_count').sum().alias('total_words'),
            pl.col('char_count').sum().alias('total_chars'),
            pl.col('unique_word_count').sum().alias('total_unique_words'),
            pl.col('punctuation_count').sum().alias('total_punctuation'),
            pl.col('conjunction_count').sum().alias('total_conjunctions'),
            
            # 分布特征
            pl.col('word_count').mean().alias('avg_sentence_length'),
            pl.col('word_count').std().alias('sentence_length_std'),
            pl.col('word_count').min().alias('min_sentence_length'),
            pl.col('word_count').max().alias('max_sentence_length'),
            
            # 词汇多样性
            (pl.col('unique_word_count').sum() / pl.col('word_count').sum()).alias('vocabulary_diversity'),
            
            # 复杂度指标
            (pl.col('punctuation_count').sum() / pl.col('sentence_count')).alias('avg_punctuation_per_sentence'),
            (pl.col('conjunction_count').sum() / pl.col('sentence_count')).alias('avg_conjunction_per_sentence')
        ])
        
        return doc_features
    
    def calculate_approximate_word_entropy(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        近似词汇熵计算
        使用哈希桶避免存储所有词汇
        """
        print("计算近似词汇熵...")
        
        # 按qid和filing_date分组处理
        def process_group(group_df: pl.DataFrame) -> pl.DataFrame:
            # 提取所有文本
            texts = group_df['clean_text'].drop_nulls().to_list()
            
            if not texts:
                return group_df.select(['qid', 'filing_date']).unique().with_columns([
                    pl.lit(0.0).alias('word_entropy_approx')
                ])
            
            # 使用哈希桶统计词频
            word_buckets = Counter()
            total_words = 0
            
            for text in texts:
                if text and isinstance(text, str):
                    words = text.split()
                    for word in words:
                        if word:
                            bucket = self._hash_word(word)
                            word_buckets[bucket] += 1
                            total_words += 1
            
            if total_words == 0:
                entropy_val = 0.0
            else:
                # 计算桶的概率分布
                probs = np.array(list(word_buckets.values())) / total_words
                entropy_val = entropy(probs, base=2)
            
            return group_df.select(['qid', 'filing_date']).unique().with_columns([
                pl.lit(entropy_val).alias('word_entropy_approx')
            ])
        
        # 分组计算
        return df.group_by(['qid', 'filing_date']).map_groups(process_group)
    
    def calculate_filing_level_features(self, doc_features: pl.DataFrame) -> pl.DataFrame:
        """
        计算filing级别特征
        """
        print("计算filing级别特征...")
        
        filing_features = doc_features.group_by(['qid', 'filing_date']).agg([
            # 文档数量和分布
            pl.len().alias('document_count'),
            pl.col('sentence_count').sum().alias('total_sentences'),
            pl.col('total_words').sum().alias('filing_total_words'),
            pl.col('total_chars').sum().alias('filing_total_chars'),
            
            # 文档大小分布熵
            pl.col('sentence_count').map_elements(
                lambda x: entropy(x.to_numpy(), base=2) if len(x) > 1 else 0.0,
                return_dtype=pl.Float64
            ).alias('document_size_entropy'),
            
            # 词汇多样性聚合
            pl.col('vocabulary_diversity').mean().alias('avg_vocabulary_diversity'),
            pl.col('vocabulary_diversity').std().alias('vocabulary_diversity_std'),
            
            # 复杂度聚合
            pl.col('avg_punctuation_per_sentence').mean().alias('filing_punctuation_complexity'),
            pl.col('avg_conjunction_per_sentence').mean().alias('filing_conjunction_complexity'),
            
            # 句子长度分布
            pl.col('avg_sentence_length').mean().alias('filing_avg_sentence_length'),
            pl.col('sentence_length_std').mean().alias('filing_sentence_length_variability')
        ])
        
        return filing_features
    
    def calculate_temporal_features_streaming(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        流式计算时间特征
        使用窗口函数避免大量数据join
        """
        print("计算时间特征...")
        
        # 确保数据按时间排序
        df_sorted = df.sort(['qid', 'filing_date'])
        
        # 使用窗口函数计算时间特征
        temporal_features = df_sorted.with_columns([
            # 滚动窗口统计（不需要显式join）
            pl.col('document_size_entropy').rolling_mean(
                window_size=f"{self.window_days}d",
                by='filing_date'
            ).over('qid').alias('entropy_rolling_mean'),
            
            pl.col('document_size_entropy').rolling_std(
                window_size=f"{self.window_days}d",
                by='filing_date'
            ).over('qid').alias('entropy_rolling_std'),
            
            pl.col('avg_vocabulary_diversity').rolling_mean(
                window_size=f"{self.window_days}d",
                by='filing_date'
            ).over('qid').alias('diversity_rolling_mean'),
            
            # 变化率
            pl.col('document_size_entropy').pct_change().over('qid').alias('entropy_change_rate'),
            pl.col('avg_vocabulary_diversity').pct_change().over('qid').alias('diversity_change_rate'),
            pl.col('filing_total_words').pct_change().over('qid').alias('volume_change_rate'),
            
            # 排名特征
            pl.col('document_size_entropy').rank().over('filing_date').alias('entropy_cross_rank'),
            pl.col('avg_vocabulary_diversity').rank().over('filing_date').alias('diversity_cross_rank'),
            
            # 时间趋势
            pl.col('document_size_entropy').diff().over('qid').alias('entropy_momentum'),
            pl.col('avg_vocabulary_diversity').diff().over('qid').alias('diversity_momentum')
        ])
        
        return temporal_features
    
    def create_robust_composite_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        创建鲁棒的复合因子
        """
        print("创建复合因子...")
        
        return df.with_columns([
            # 信息复杂度因子
            (pl.col('document_size_entropy') * 0.3 + 
             pl.col('avg_vocabulary_diversity') * 0.4 + 
             pl.col('filing_punctuation_complexity') * 0.3).alias('information_complexity'),
            
            # 信息变化因子
            (pl.col('entropy_change_rate').abs() * 0.5 + 
             pl.col('diversity_change_rate').abs() * 0.5).alias('information_change_intensity'),
            
            # 信息异常因子
            (pl.col('document_size_entropy') - pl.col('entropy_rolling_mean')).abs() / 
            (pl.col('entropy_rolling_std') + 1e-10).alias('information_anomaly'),
            
            # 信息密度因子
            (pl.col('filing_total_words') / (pl.col('document_count') + 1)).alias('information_density'),
            
            # 信息动量因子
            (pl.col('entropy_momentum') * 0.5 + 
             pl.col('diversity_momentum') * 0.5).alias('information_momentum'),
            
            # 相对信息位置
            (pl.col('entropy_cross_rank') / pl.col('entropy_cross_rank').max().over('filing_date')).alias('relative_complexity_position'),
            
            # 信息波动率
            pl.col('entropy_rolling_std').alias('information_volatility')
        ])
    
    def process_in_chunks(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        分块处理主函数
        """
        print(f"开始分块处理，数据总量: {len(df):,} 行")
        
        # 检查是否有缓存
        cache_key = f"processed_factors_{len(df)}_{hash(str(df.columns))}"
        cached_result = self._load_cache(cache_key)
        if cached_result is not None:
            print("从缓存加载结果")
            return cached_result
        
        # 获取数据的日期范围，用于优化分块策略
        date_range = df.select([
            pl.col('filing_date').min().alias('min_date'),
            pl.col('filing_date').max().alias('max_date')
        ]).row(0)
        
        print(f"数据日期范围: {date_range[0]} 到 {date_range[1]}")
        
        # 按qid分块，避免跨股票的复杂计算
        unique_qids = df.select('qid').unique().to_series().to_list()
        total_qids = len(unique_qids)
        
        print(f"总计股票数: {total_qids}")
        
        # 分批处理股票
        batch_size = max(1, self.chunk_size // 100000)  # 根据数据量调整批大小
        all_results = []
        
        for i in range(0, total_qids, batch_size):
            batch_qids = unique_qids[i:i + batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(total_qids-1)//batch_size + 1}, 股票数: {len(batch_qids)}")
            
            # 提取当前批次数据
            batch_df = df.filter(pl.col('qid').is_in(batch_qids))
            
            # 处理当前批次
            batch_result = self._process_batch(batch_df)
            all_results.append(batch_result)
            
            # 强制垃圾回收
            del batch_df
            gc.collect()
        
        # 合并所有结果
        print("合并所有批次结果...")
        final_result = pl.concat(all_results)
        
        # 最终标准化
        final_result = self._final_normalization(final_result)
        
        # 保存缓存
        self._save_cache(final_result, cache_key)
        
        return final_result
    
    def _process_batch(self, batch_df: pl.DataFrame) -> pl.DataFrame:
        """处理单个批次"""
        # 1. 文本预处理
        processed = self.preprocess_text_streaming(batch_df)
        
        # 2. 文档级别特征
        doc_features = self.calculate_document_level_features(processed)
        
        # 3. Filing级别特征
        filing_features = self.calculate_filing_level_features(doc_features)
        
        # 4. 近似词汇熵
        word_entropy = self.calculate_approximate_word_entropy(processed)
        
        # 5. 合并特征
        combined_features = filing_features.join(
            word_entropy, on=['qid', 'filing_date'], how='left'
        ).fill_null(0.0)
        
        # 6. 时间特征
        temporal_features = self.calculate_temporal_features_streaming(combined_features)
        
        # 7. 复合因子
        final_features = self.create_robust_composite_factors(temporal_features)
        
        return final_features
    
    def _final_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
        """最终标准化"""
        print("执行最终标准化...")
        
        factor_cols = [
            'information_complexity', 'information_change_intensity', 
            'information_anomaly', 'information_density', 'information_momentum',
            'relative_complexity_position', 'information_volatility'
        ]
        
        # 按日期截面标准化
        expressions = []
        for col in factor_cols:
            expressions.extend([
                # 分位数排名
                (pl.col(col).rank() / pl.col(col).count()).over('filing_date').alias(f'{col}_rank'),
                # 稳健Z-score（使用中位数和MAD）
                ((pl.col(col) - pl.col(col).median().over('filing_date')) / 
                 (pl.col(col).mad().over('filing_date') + 1e-10)).alias(f'{col}_robust_zscore')
            ])
        
        return df.with_columns(expressions)

def demo_scalable_usage():
    """演示可扩展版本的用法"""
    print("=== 超大规模数据处理演示 ===")
    
    # 创建更大的模拟数据集
    np.random.seed(42)
    
    # 模拟1000万行数据
    print("生成大规模模拟数据...")
    dates = pl.date_range(
        start=pl.date(2020, 1, 1),
        end=pl.date(2024, 1, 1),
        interval='1d'
    )
    
    qids = [f'QID_{i:04d}' for i in range(1, 1001)]  # 1000只股票
    
    # 分批生成数据避免内存溢出
    batch_size = 100
    all_data = []
    
    for batch_start in range(0, len(qids), batch_size):
        batch_qids = qids[batch_start:batch_start + batch_size]
        print(f"生成数据批次: {batch_start//batch_size + 1}/{len(qids)//batch_size + 1}")
        
        batch_data = []
        for qid in batch_qids:
            # 每只股票随机选择filing日期
            n_filings = np.random.randint(200, 500)
            filing_dates = np.random.choice(dates, size=n_filings, replace=False)
            
            for filing_date in filing_dates:
                doc_id = f'DOC_{qid}_{filing_date}'
                num_sentences = np.random.randint(20, 100)
                
                for sent_id in range(num_sentences):
                    sentence_templates = [
                        f"During {np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {np.random.randint(2020, 2024)}, the company achieved {np.random.choice(['record', 'strong', 'stable', 'improved'])} financial performance with revenue of ${np.random.randint(100, 10000)} million.",
                        f"The board of directors has approved a comprehensive strategic plan to {np.random.choice(['expand', 'optimize', 'restructure', 'enhance'])} our {np.random.choice(['operations', 'market presence', 'technology platform', 'customer base'])} over the next {np.random.randint(1, 5)} years.",
                        f"Management announced {np.random.choice(['significant', 'strategic', 'transformative', 'innovative'])} initiatives including {np.random.choice(['digital transformation', 'market expansion', 'cost optimization', 'product development'])} programs expected to generate ${np.random.randint(50, 500)} million in {np.random.choice(['revenue', 'savings', 'value'])}.",
                        f"The company continues to navigate {np.random.choice(['challenging', 'dynamic', 'evolving', 'competitive'])} market conditions while maintaining {np.random.choice(['strong', 'resilient', 'stable', 'robust'])} operational performance and {np.random.choice(['solid', 'healthy', 'improving', 'strong'])} financial position.",
                        f"Regulatory developments and {np.random.choice(['compliance', 'environmental', 'safety', 'governance'])} requirements continue to shape our {np.random.choice(['business strategy', 'operational approach', 'investment priorities', 'risk management'])} as we work to {np.random.choice(['exceed', 'meet', 'maintain', 'enhance'])} all applicable standards."
                    ]
                    
                    sentence = np.random.choice(sentence_templates)
                    
                    batch_data.append({
                        'qid': qid,
                        'document_id': doc_id,
                        'sentence_id': f'{doc_id}_S{sent_id}',
                        'sentence_text': sentence,
                        'filing_date': filing_date
                    })
        
        all_data.extend(batch_data)
    
    print(f"总数据量: {len(all_data):,} 行")
    df = pl.DataFrame(all_data)
    
    # 创建可扩展引擎
    engine = ScalableEntropyFactorEngine(
        window_days=90,
        chunk_size=500000,  # 50万行一个块
        n_vocab_buckets=5000,  # 减少哈希桶数量
        enable_disk_cache=True
    )
    
    # 处理数据
    import time
    start_time = time.time()
    
    print("开始处理数据...")
    try:
        factors = engine.process_in_chunks(df)
        
        end_time = time.time()
        print(f"处理完成！耗时: {end_time - start_time:.2f} 秒")
        
        print(f"结果维度: {factors.shape}")
        print("\n关键因子统计:")
        print(factors.select([
            'information_complexity', 'information_change_intensity', 
            'information_anomaly', 'information_density', 'information_momentum'
        ]).describe())
        
        # 显示内存使用情况
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\n内存使用: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")
        
        return factors
        
    except Exception as e:
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    factors = demo_scalable_usage()
    
    if factors is not None:
        print("\n=== 超大规模优化策略 ===")
        print("1. 分块处理: 避免全量数据加载")
        print("2. 流式计算: 减少内存峰值")
        print("3. 磁盘缓存: 支持断点续传")
        print("4. 哈希桶: 控制词汇空间大小")
        print("5. 近似算法: 牺牲精度换取性能")
        print("6. 垃圾回收: 及时释放内存")
        print("7. 窗口函数: 避免大表join")
        print("8. 批次处理: 并行化友好")
