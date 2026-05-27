"""路由分层评估器和α桶分析

实现Engram风格的路由分层评估和α桶分析，用于评估Δ胶囊的检索性能。
严格遵循Pro建议：无PyTorch依赖，纯Python实现。

用法：
    from openllm_memory.capsule.retrieval_evaluator import RetrievalEvaluator, AlphaBucketAnalyzer
    
    evaluator = RetrievalEvaluator(hot_keys=["user.name", "user.age"])
    results = evaluator.evaluate(search_results, query)
    
    analyzer = AlphaBucketAnalyzer()
    stats = analyzer.analyze(weights, scores)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple


class RetrievalEvaluator:
    """路由分层评估器
    
    实现Hot/Cold分层评估，用于分析检索性能。
    
    Attributes:
        hot_keys: 热键集合
    """
    
    def __init__(self, hot_keys: Optional[List[str]] = None):
        """初始化路由分层评估器
        
        Args:
            hot_keys: 热键列表，这些键被认为是高频或重要的
        """
        self.hot_keys = set(hot_keys) if hot_keys else set()
    
    def evaluate(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """路由分层评估
        
        Args:
            results: 检索结果列表，每个结果包含 'key', 'value', 'score' 等字段
            query: 查询字符串
            
        Returns:
            评估结果字典
        """
        hot_matches = []
        cold_matches = []
        
        for result in results:
            score = self._compute_score(result, query)
            key = result.get('key', '')
            
            if key in self.hot_keys:
                hot_matches.append((score, result))
            else:
                cold_matches.append((score, result))
        
        # 计算统计
        hot_scores = [s for s, _ in hot_matches]
        cold_scores = [s for s, _ in cold_matches]
        
        hot_mean = sum(hot_scores) / len(hot_scores) if hot_scores else 0
        cold_mean = sum(cold_scores) / len(cold_scores) if cold_scores else 0
        hot_cold_delta = hot_mean - cold_mean if hot_scores and cold_scores else 0
        
        return {
            'hot_mean': hot_mean,
            'cold_mean': cold_mean,
            'hot_cold_delta': hot_cold_delta,
            'hot_count': len(hot_matches),
            'cold_count': len(cold_matches),
            'total_count': len(results),
            'hot_ratio': len(hot_matches) / len(results) if results else 0,
        }
    
    def _compute_score(self, result: Dict[str, Any], query: str) -> float:
        """计算匹配分数
        
        Args:
            result: 检索结果
            query: 查询字符串
            
        Returns:
            匹配分数
        """
        score = 0.0
        query_lower = query.lower()
        
        # 检查key匹配
        key = str(result.get('key', '')).lower()
        if query_lower in key:
            score += 3.0
        
        # 检查value匹配
        value = result.get('value', '')
        if isinstance(value, str) and query_lower in value.lower():
            score += 1.0
        
        # 检查是否有额外的score字段
        if 'score' in result:
            score += result['score']
        
        return score
    
    def add_hot_key(self, key: str) -> None:
        """添加热键
        
        Args:
            key: 要添加的热键
        """
        self.hot_keys.add(key)
    
    def remove_hot_key(self, key: str) -> None:
        """移除热键
        
        Args:
            key: 要移除的热键
        """
        self.hot_keys.discard(key)
    
    def set_hot_keys(self, hot_keys: List[str]) -> None:
        """设置热键列表
        
        Args:
            hot_keys: 热键列表
        """
        self.hot_keys = set(hot_keys)
    
    def get_hot_keys(self) -> List[str]:
        """获取热键列表
        
        Returns:
            热键列表
        """
        return list(self.hot_keys)


class AlphaBucketAnalyzer:
    """α桶分析器
    
    实现按权重分桶的分析，用于评估门控机制的效果。
    
    Attributes:
        buckets: 桶边界列表
    """
    
    def __init__(self, buckets: Optional[List[float]] = None):
        """初始化α桶分析器
        
        Args:
            buckets: 桶边界列表，默认为 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        """
        self.buckets = buckets or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.bucket_stats: Dict[int, List[float]] = {i: [] for i in range(len(self.buckets) - 1)}
    
    def analyze(self, weights: List[float], scores: List[float]) -> Dict[str, Any]:
        """分析权重与分数的关系
        
        Args:
            weights: 权重列表
            scores: 分数列表
            
        Returns:
            分析结果字典
        """
        if len(weights) != len(scores):
            raise ValueError("weights and scores must have the same length")
        
        # 清空之前的统计
        self.bucket_stats = {i: [] for i in range(len(self.buckets) - 1)}
        
        # 将分数分配到对应的桶
        for weight, score in zip(weights, scores):
            bucket_idx = self._get_bucket(weight)
            self.bucket_stats[bucket_idx].append(score)
        
        # 计算各桶统计
        results = {}
        for idx in range(len(self.buckets) - 1):
            bucket_scores = self.bucket_stats[idx]
            bucket_name = f"{self.buckets[idx]:.1f}-{self.buckets[idx+1]:.1f}"
            
            if bucket_scores:
                results[bucket_name] = {
                    'mean_score': sum(bucket_scores) / len(bucket_scores),
                    'count': len(bucket_scores),
                    'min_score': min(bucket_scores),
                    'max_score': max(bucket_scores),
                    'std_score': self._std(bucket_scores),
                }
            else:
                results[bucket_name] = {
                    'mean_score': 0.0,
                    'count': 0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'std_score': 0.0,
                }
        
        # 计算总体统计
        all_scores = scores
        results['overall'] = {
            'mean_score': sum(all_scores) / len(all_scores) if all_scores else 0.0,
            'count': len(all_scores),
            'min_score': min(all_scores) if all_scores else 0.0,
            'max_score': max(all_scores) if all_scores else 0.0,
            'std_score': self._std(all_scores),
        }
        
        return results
    
    def _get_bucket(self, weight: float) -> int:
        """获取权重所在的桶索引
        
        Args:
            weight: 权重值
            
        Returns:
            桶索引
        """
        for i in range(len(self.buckets) - 1):
            if self.buckets[i] <= weight < self.buckets[i+1]:
                return i
        # 如果权重等于最后一个桶的上界，放在最后一个桶
        return len(self.buckets) - 2
    
    def _std(self, values: List[float]) -> float:
        """计算标准差
        
        Args:
            values: 数值列表
            
        Returns:
            标准差
        """
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def get_bucket_boundaries(self) -> List[float]:
        """获取桶边界
        
        Returns:
            桶边界列表
        """
        return self.buckets.copy()
    
    def set_bucket_boundaries(self, buckets: List[float]) -> None:
        """设置桶边界
        
        Args:
            buckets: 桶边界列表
        """
        self.buckets = buckets
        self.bucket_stats = {i: [] for i in range(len(self.buckets) - 1)}


class RetrievalEvaluatorWithFallback:
    """带回滚机制的路由分层评估器
    
    支持在详细评估和简单统计之间切换。
    
    Attributes:
        evaluator: 路由分层评估器实例
        use_detailed_evaluation: 是否使用详细评估
    """
    
    def __init__(self, hot_keys: Optional[List[str]] = None, use_detailed_evaluation: bool = True):
        """初始化带回滚机制的路由分层评估器
        
        Args:
            hot_keys: 热键列表
            use_detailed_evaluation: 是否使用详细评估
        """
        self.evaluator = RetrievalEvaluator(hot_keys)
        self.use_detailed_evaluation = use_detailed_evaluation
    
    def evaluate(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """评估检索结果
        
        Args:
            results: 检索结果列表
            query: 查询字符串
            
        Returns:
            评估结果字典
        """
        if self.use_detailed_evaluation:
            return self.evaluator.evaluate(results, query)
        else:
            # 回滚到简单统计
            return self._simple_evaluate(results, query)
    
    def _simple_evaluate(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """简单评估（回滚方案）
        
        Args:
            results: 检索结果列表
            query: 查询字符串
            
        Returns:
            简单统计结果
        """
        scores = []
        for result in results:
            score = self.evaluator._compute_score(result, query)
            scores.append(score)
        
        mean_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'mean_score': mean_score,
            'count': len(results),
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'hot_count': 0,  # 简单评估不区分热冷
            'cold_count': len(results),
            'total_count': len(results),
            'hot_ratio': 0.0,
        }
    
    def enable_detailed_evaluation(self) -> None:
        """启用详细评估"""
        self.use_detailed_evaluation = True
    
    def disable_detailed_evaluation(self) -> None:
        """禁用详细评估（回滚到简单统计）"""
        self.use_detailed_evaluation = False
    
    def add_hot_key(self, key: str) -> None:
        """添加热键
        
        Args:
            key: 要添加的热键
        """
        self.evaluator.add_hot_key(key)
    
    def remove_hot_key(self, key: str) -> None:
        """移除热键
        
        Args:
            key: 要移除的热键
        """
        self.evaluator.remove_hot_key(key)
    
    def set_hot_keys(self, hot_keys: List[str]) -> None:
        """设置热键列表
        
        Args:
            hot_keys: 热键列表
        """
        self.evaluator.set_hot_keys(hot_keys)
    
    def get_hot_keys(self) -> List[str]:
        """获取热键列表
        
        Returns:
            热键列表
        """
        return self.evaluator.get_hot_keys()