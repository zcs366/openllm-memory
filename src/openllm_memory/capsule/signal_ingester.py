"""信号入库管道——将提取的信号写入Δ胶囊

流程：
1. 检查是否与现有记忆重复（用SimilarityGate）
2. 计算信号的存储key
3. 写入Δ胶囊（op=set，带元数据）

这是Δ胶囊的延迟信号通道的最后一步——将隐式信号转化为持久记忆。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .core import Capsule
from .session_scanner import Signal
from .similarity_gate import SimilarityGate

logger = logging.getLogger(__name__)


class SignalIngester:
    """信号入库管道
    
    用法：
        capsule = Capsule.open("~/.openllm/capsule")
        gate = SimilarityGate()
        ingester = SignalIngester(capsule, gate)
        
        signals = scanner.scan(days=7)
        written = ingester.ingest(signals)
        print(f"Ingested {written}/{len(signals)} signals")
    """
    
    def __init__(self, capsule: Capsule, gate: Optional[SimilarityGate] = None):
        self.capsule = capsule
        self.gate = gate or SimilarityGate()
        self._stats = {
            "total": 0,
            "written": 0,
            "skipped_duplicate": 0,
            "skipped_low_confidence": 0,
            "errors": 0,
        }
    
    def ingest(self, signals: List[Signal], 
               min_confidence: float = 0.3,
               dedup_threshold: float = 0.8) -> int:
        """批量入库信号
        
        Args:
            signals: 信号列表
            min_confidence: 最小置信度阈值
            dedup_threshold: 去重相似度阈值（>此值视为重复）
        
        Returns:
            实际写入的信号数量
        """
        self._stats["total"] += len(signals)
        written = 0
        
        for signal in signals:
            try:
                # 1. 检查置信度
                if signal.confidence < min_confidence:
                    self._stats["skipped_low_confidence"] += 1
                    continue
                
                # 2. 检查是否重复
                if self._is_duplicate(signal, dedup_threshold):
                    self._stats["skipped_duplicate"] += 1
                    continue
                
                # 3. 写入Δ胶囊
                self._write_signal(signal)
                written += 1
                self._stats["written"] += 1
                
            except Exception as e:
                logger.error("Failed to ingest signal %s: %s", signal.key, e)
                self._stats["errors"] += 1
        
        logger.info("Ingested %d/%d signals (dup=%d, low_conf=%d, err=%d)",
                     written, len(signals),
                     self._stats["skipped_duplicate"],
                     self._stats["skipped_low_confidence"],
                     self._stats["errors"])
        
        return written
    
    def _is_duplicate(self, signal: Signal, threshold: float) -> bool:
        """检查信号是否与现有记忆重复"""
        # 用prefetch查找相关记忆
        existing = self.capsule.prefetch(signal.key, max_results=3)
        if not existing:
            return False
        
        # 解析existing内容
        existing_values = self._parse_prefetch_result(existing)
        if not existing_values:
            return False
        
        # 用SimilarityGate逐个比较
        try:
            signal_value_str = str(signal.value)
            for existing_value in existing_values:
                similarity = self.gate.compute_similarity(
                    signal_value_str,
                    existing_value,
                )
                if similarity > threshold:
                    return True
            return False
        except Exception as e:
            logger.warning("Similarity check failed: %s", e)
            return False
    
    def _parse_prefetch_result(self, prefetch_text: str) -> List[str]:
        """解析prefetch返回的文本，提取value列表"""
        values = []
        for line in prefetch_text.split("\n"):
            if "=" in line:
                # 格式：- [timestamp] op key = value
                parts = line.split("=", 1)
                if len(parts) == 2:
                    value = parts[1].strip()
                    if value:
                        values.append(value)
        return values
    
    def _write_signal(self, signal: Signal) -> None:
        """将信号写入Δ胶囊"""
        # 构建存储key
        key = self._build_key(signal)
        
        # 构建存储value
        value = {
            "value": signal.value,
            "confidence": signal.confidence,
            "source_session": signal.source_session,
            "source_timestamp": signal.source_timestamp,
            "extracted_at": time.time(),
            "raw_content": signal.raw_content[:200],
        }
        
        # 写入Δ胶囊
        self.capsule.write(
            key=key,
            value=value,
            op="set",
            session_id=signal.source_session,
            metadata={
                "signal_type": signal.type,
                "auto_extracted": True,
                "confidence": signal.confidence,
            },
        )
        
        logger.debug("Wrote signal: %s = %s", key, signal.value)
    
    def _build_key(self, signal: Signal) -> str:
        """构建存储key
        
        格式：signal/{type}/{key}
        例如：signal/preference/user.style
        """
        # 清理key中的特殊字符
        clean_key = signal.key.replace(".", "_").replace("/", "_")
        return f"signal/{signal.type}/{clean_key}"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "total": 0,
            "written": 0,
            "skipped_duplicate": 0,
            "skipped_low_confidence": 0,
            "errors": 0,
        }


class SignalIngesterWithFallback:
    """带回滚的信号入库管道
    
    如果SimilarityGate失败，回退到简单字符串匹配去重。
    """
    
    def __init__(self, capsule: Capsule, use_similarity_gate: bool = True):
        self.capsule = capsule
        self.use_similarity_gate = use_similarity_gate
        
        if use_similarity_gate:
            try:
                self.gate = SimilarityGate()
                self._gate_available = True
            except Exception as e:
                logger.warning("SimilarityGate init failed, using fallback: %s", e)
                self._gate_available = False
        else:
            self._gate_available = False
        
        self._stats = {
            "total": 0,
            "written": 0,
            "skipped_duplicate": 0,
            "skipped_low_confidence": 0,
            "errors": 0,
            "fallback_used": 0,
        }
    
    def ingest(self, signals: List[Signal], 
               min_confidence: float = 0.3) -> int:
        """批量入库信号（带回滚）"""
        self._stats["total"] += len(signals)
        written = 0
        
        for signal in signals:
            try:
                # 1. 检查置信度
                if signal.confidence < min_confidence:
                    self._stats["skipped_low_confidence"] += 1
                    continue
                
                # 2. 检查是否重复
                if self._is_duplicate_fallback(signal):
                    self._stats["skipped_duplicate"] += 1
                    continue
                
                # 3. 写入Δ胶囊
                self._write_signal(signal)
                written += 1
                self._stats["written"] += 1
                
            except Exception as e:
                logger.error("Failed to ingest signal %s: %s", signal.key, e)
                self._stats["errors"] += 1
        
        return written
    
    def _is_duplicate_fallback(self, signal: Signal) -> bool:
        """回退的去重方法：简单字符串匹配"""
        existing = self.capsule.prefetch(signal.key, max_results=3)
        if not existing:
            return False
        
        # 简单字符串匹配
        signal_value_str = str(signal.value).lower()
        for line in existing.split("\n"):
            if "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    existing_value = parts[1].strip().lower()
                    # 如果value完全相同或包含，视为重复
                    if signal_value_str == existing_value or signal_value_str in existing_value:
                        self._stats["fallback_used"] += 1
                        return True
        
        return False
    
    def _write_signal(self, signal: Signal) -> None:
        """将信号写入Δ胶囊"""
        clean_key = signal.key.replace(".", "_").replace("/", "_")
        key = f"signal/{signal.type}/{clean_key}"
        
        value = {
            "value": signal.value,
            "confidence": signal.confidence,
            "source_session": signal.source_session,
            "source_timestamp": signal.source_timestamp,
            "extracted_at": time.time(),
            "raw_content": signal.raw_content[:200],
        }
        
        self.capsule.write(
            key=key,
            value=value,
            op="set",
            session_id=signal.source_session,
            metadata={
                "signal_type": signal.type,
                "auto_extracted": True,
                "confidence": signal.confidence,
            },
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
