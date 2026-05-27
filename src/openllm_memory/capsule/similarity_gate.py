"""基于TF-IDF的相似度门控（无PyTorch依赖）

实现查询-记忆相似度门控，用于Δ胶囊的记忆检索。
严格遵循Pro建议：无PyTorch依赖，使用TF-IDF和余弦相似度。

用法：
    from openllm_memory.capsule.similarity_gate import SimilarityGate
    
    gate = SimilarityGate()
    gate.fit(["user.name", "user.age", "project.status"])
    weights = gate.compute_weights("user", ["张成市", "30", "active"])
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class SimilarityGate:
    """基于TF-IDF的相似度门控
    
    实现查询与记忆的相似度计算，用于权重分配。
    无PyTorch依赖，使用纯Python实现。
    
    Attributes:
        vocabulary: 词汇表
        idf: IDF值
        fitted: 是否已训练
    """
    
    def __init__(self):
        """初始化相似度门控"""
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False
        self.documents: List[str] = []
    
    def fit(self, documents: List[str]) -> None:
        """训练TF-IDF向量化器
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        
        # 构建词汇表
        self.vocabulary = {}
        word_doc_count = {}
        
        for doc_idx, doc in enumerate(documents):
            # 分词（简单实现：按空格和标点分割）
            words = self._tokenize(doc)
            unique_words = set(words)
            
            for word in unique_words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # 计算IDF
        num_docs = len(documents)
        self.idf = {}
        for word, doc_count in word_doc_count.items():
            self.idf[word] = math.log(num_docs / (1 + doc_count))
        
        self.fitted = True
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词器
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        import re
        # 转换为小写，按非字母数字字符分割
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _compute_tf(self, text: str) -> Dict[str, float]:
        """计算词频（TF）
        
        Args:
            text: 输入文本
            
        Returns:
            词频字典
        """
        words = self._tokenize(text)
        word_count = {}
        
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 归一化
        total_words = len(words)
        tf = {}
        for word, count in word_count.items():
            tf[word] = count / total_words if total_words > 0 else 0
        
        return tf
    
    def _compute_tfidf(self, text: str) -> Dict[str, float]:
        """计算TF-IDF向量
        
        Args:
            text: 输入文本
            
        Returns:
            TF-IDF向量
        """
        tf = self._compute_tf(text)
        tfidf = {}
        
        for word, tf_value in tf.items():
            if word in self.idf:
                tfidf[word] = tf_value * self.idf[word]
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        # 获取共同键
        common_keys = set(vec1.keys()) & set(vec2.keys())
        
        if not common_keys:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        
        # 计算范数
        norm1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
        norm2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_similarity(self, query: str, document: str) -> float:
        """计算查询与文档的相似度
        
        Args:
            query: 查询字符串
            document: 文档字符串
            
        Returns:
            相似度分数
        """
        if not self.fitted:
            raise ValueError("SimilarityGate not fitted. Call fit() first.")
        
        # 计算TF-IDF向量
        query_tfidf = self._compute_tfidf(query)
        doc_tfidf = self._compute_tfidf(document)
        
        # 计算余弦相似度
        return self._cosine_similarity(query_tfidf, doc_tfidf)
    
    def compute_weights(self, query: str, documents: List[str], temperature: float = 1.0) -> List[float]:
        """计算查询与多个文档的相似度权重
        
        Args:
            query: 查询字符串
            documents: 文档列表
            temperature: 温度参数，控制权重的平滑程度
            
        Returns:
            权重列表
        """
        if not self.fitted:
            raise ValueError("SimilarityGate not fitted. Call fit() first.")
        
        # 计算相似度
        similarities = []
        for doc in documents:
            sim = self.compute_similarity(query, doc)
            similarities.append(sim)
        
        # 应用温度缩放
        if temperature != 1.0:
            similarities = [s / temperature for s in similarities]
        
        # 归一化（softmax）
        max_sim = max(similarities) if similarities else 0
        exp_sims = [math.exp(s - max_sim) for s in similarities]  # 数值稳定性
        sum_exp_sims = sum(exp_sims)
        
        if sum_exp_sims == 0:
            # 如果所有相似度都是0，均匀分布
            weights = [1.0 / len(documents)] * len(documents)
        else:
            weights = [exp_sim / sum_exp_sims for exp_sim in exp_sims]
        
        return weights
    
    def compute_weights_with_scores(self, query: str, documents: List[str], temperature: float = 1.0) -> List[Tuple[float, str]]:
        """计算查询与多个文档的相似度权重（带分数）
        
        Args:
            query: 查询字符串
            documents: 文档列表
            temperature: 温度参数
            
        Returns:
            (权重, 文档)元组列表
        """
        weights = self.compute_weights(query, documents, temperature)
        return list(zip(weights, documents))
    
    def get_vocabulary(self) -> Dict[str, int]:
        """获取词汇表
        
        Returns:
            词汇表字典
        """
        return self.vocabulary.copy()
    
    def get_idf(self) -> Dict[str, float]:
        """获取IDF值
        
        Returns:
            IDF字典
        """
        return self.idf.copy()
    
    def is_fitted(self) -> bool:
        """是否已训练
        
        Returns:
            是否已训练
        """
        return self.fitted


class SimilarityGateWithFallback:
    """带回滚机制的相似度门控
    
    支持在TF-IDF门控和简单关键词匹配之间切换。
    
    Attributes:
        similarity_gate: 相似度门控实例
        use_similarity_gate: 是否使用相似度门控
    """
    
    def __init__(self, use_similarity_gate: bool = True):
        """初始化带回滚机制的相似度门控
        
        Args:
            use_similarity_gate: 是否使用相似度门控
        """
        self.similarity_gate = SimilarityGate()
        self.use_similarity_gate = use_similarity_gate
        self._fitted = False
    
    def fit(self, documents: List[str]) -> None:
        """训练相似度门控
        
        Args:
            documents: 文档列表
        """
        if self.use_similarity_gate:
            self.similarity_gate.fit(documents)
            self._fitted = True
    
    def compute_weights(self, query: str, documents: List[str], temperature: float = 1.0) -> List[float]:
        """计算权重
        
        Args:
            query: 查询字符串
            documents: 文档列表
            temperature: 温度参数
            
        Returns:
            权重列表
        """
        if self.use_similarity_gate and self._fitted:
            return self.similarity_gate.compute_weights(query, documents, temperature)
        else:
            # 回滚到简单关键词匹配
            weights = []
            query_lower = query.lower()
            
            for doc in documents:
                # 简单匹配：检查查询是否在文档中
                if query_lower in doc.lower():
                    weights.append(1.0)
                else:
                    weights.append(0.0)
            
            # 归一化
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                # 如果没有匹配，均匀分布
                weights = [1.0 / len(documents)] * len(documents)
            
            return weights
    
    def compute_weights_with_scores(self, query: str, documents: List[str], temperature: float = 1.0) -> List[Tuple[float, str]]:
        """计算权重（带分数）
        
        Args:
            query: 查询字符串
            documents: 文档列表
            temperature: 温度参数
            
        Returns:
            (权重, 文档)元组列表
        """
        weights = self.compute_weights(query, documents, temperature)
        return list(zip(weights, documents))
    
    def enable_similarity_gate(self) -> None:
        """启用相似度门控"""
        self.use_similarity_gate = True
    
    def disable_similarity_gate(self) -> None:
        """禁用相似度门控（回滚到简单匹配）"""
        self.use_similarity_gate = False
    
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._fitted