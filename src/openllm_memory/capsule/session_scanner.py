"""Session Scanner引擎——从Hermes session DB中挖掘隐式信号

扫描state.db的messages表，用FTS5全文搜索找关键词，
提取用户纠正、偏好、决策、模式等信号。

这是Δ胶囊的延迟信号通道——捕捉用户没有显式说"记住"但隐式表达的信息。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .signal_patterns import SignalPatterns

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """一个提取的信号"""
    type: str  # correction/preference/decision/pattern
    key: str  # 信号键（如 "user.style"）
    value: Any  # 信号值（如 "concise"）
    confidence: float  # 置信度 (0.0-1.0)
    source_session: str  # 来源session ID
    source_timestamp: float  # 来源消息时间戳
    raw_content: str = ""  # 原始内容（用于调试）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source_session": self.source_session,
            "source_timestamp": self.source_timestamp,
            "raw_content": self.raw_content[:200],
            "metadata": self.metadata,
        }


class SessionScanner:
    """从Hermes session DB中挖掘隐式信号
    
    用法：
        scanner = SessionScanner("~/.hermes/state.db")
        signals = scanner.scan(days=7)
        for signal in signals:
            print(f"{signal.type}: {signal.key} = {signal.value}")
    """
    
    def __init__(self, db_path: str = "~/.hermes/state.db"):
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Session DB not found: {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # 返回dict-like行
        self.patterns = SignalPatterns()
        
        # 检查FTS5索引是否存在
        self._has_fts = self._check_fts()
    
    def _check_fts(self) -> bool:
        """检查FTS5索引是否存在"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        )
        return cursor.fetchone() is not None
    
    def scan(self, days: int = 7, min_confidence: float = 0.3) -> List[Signal]:
        """扫描最近N天的session，返回发现的信号
        
        Args:
            days: 扫描最近N天
            min_confidence: 最小置信度阈值
        
        Returns:
            信号列表（按置信度降序）
        """
        cutoff = time.time() - days * 86400
        signals: List[Signal] = []
        
        # 遍历所有模式类型
        for ptype, keywords in self.patterns.all():
            for keyword in keywords:
                matches = self._search_keyword(keyword, cutoff)
                for match in matches:
                    signal = self._extract_signal(match, ptype)
                    if signal and signal.confidence >= min_confidence:
                        signals.append(signal)
        
        # 去重
        signals = self._deduplicate(signals)
        
        # 按置信度降序排序
        signals.sort(key=lambda s: -s.confidence)
        
        logger.info("Scanned %d days, found %d signals", days, len(signals))
        return signals
    
    def _search_keyword(self, keyword: str, cutoff: float) -> List[dict]:
        """用FTS5搜索关键词"""
        if not self._has_fts:
            # 回退到LIKE搜索
            return self._like_search(keyword, cutoff)
        
        try:
            # 转义FTS5特殊字符
            safe_keyword = self._escape_fts5(keyword)
            
            # FTS5搜索（只搜user消息）
            query = """
                SELECT m.id, m.session_id, m.role, m.content, m.timestamp
                FROM messages_fts fts
                JOIN messages m ON fts.rowid = m.id
                WHERE messages_fts MATCH ? 
                  AND m.timestamp > ? 
                  AND m.role = 'user'
                ORDER BY m.timestamp DESC
                LIMIT 50
            """
            cursor = self.conn.execute(query, (safe_keyword, cutoff))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning("FTS5 search failed for '%s': %s", keyword, e)
            return self._like_search(keyword, cutoff)
    
    def _escape_fts5(self, keyword: str) -> str:
        """转义FTS5特殊字符"""
        # FTS5特殊字符：* " ( ) AND OR NOT NEAR
        # 用双引号包裹整个关键词，使其成为短语搜索
        # 同时转义内部的双引号
        escaped = keyword.replace('"', '""')
        return f'"{escaped}"'
    
    def _like_search(self, keyword: str, cutoff: float) -> List[dict]:
        """LIKE搜索（回退方案）"""
        query = """
            SELECT id, session_id, role, content, timestamp
            FROM messages
            WHERE content LIKE ? 
              AND timestamp > ? 
              AND role = 'user'
            ORDER BY timestamp DESC
            LIMIT 50
        """
        cursor = self.conn.execute(query, (f"%{keyword}%", cutoff))
        return [dict(row) for row in cursor.fetchall()]
    
    def _extract_signal(self, match: dict, pattern_type: str) -> Optional[Signal]:
        """从匹配的消息中提取信号"""
        content = match.get("content", "")
        if not content:
            return None
        
        # 提取key和value
        key, value = self._extract_key_value(content, pattern_type)
        if not key or not value:
            return None
        
        # 计算置信度
        confidence = self._calculate_confidence(content, pattern_type)
        
        return Signal(
            type=pattern_type,
            key=key,
            value=value,
            confidence=confidence,
            source_session=match.get("session_id", ""),
            source_timestamp=match.get("timestamp", 0.0),
            raw_content=content[:200],
            metadata={
                "message_id": match.get("id"),
                "keyword_matched": True,
            },
        )
    
    def _extract_key_value(self, content: str, pattern_type: str) -> tuple:
        """从内容中提取key和value
        
        返回 (key, value) 或 (None, None)
        """
        content_lower = content.lower()
        
        if pattern_type == "correction":
            return self._extract_correction(content, content_lower)
        elif pattern_type == "preference":
            return self._extract_preference(content, content_lower)
        elif pattern_type == "decision":
            return self._extract_decision(content, content_lower)
        elif pattern_type == "pattern":
            return self._extract_pattern(content, content_lower)
        
        return (None, None)
    
    def _extract_correction(self, content: str, content_lower: str) -> tuple:
        """提取纠正信号
        
        模式：
        - "不对，应该是XXX" → key=correction.target, value=XXX
        - "错了，用XXX" → key=correction.tool, value=XXX
        """
        # 中文模式
        if "不对" in content or "错了" in content:
            # 找"应该是"后面的内容
            if "应该是" in content:
                idx = content.index("应该是") + len("应该是")
                value = content[idx:].strip()[:100]
                return ("correction.target", value)
            # 找"用"后面的内容
            if "用" in content:
                idx = content.index("用") + len("用")
                value = content[idx:].strip()[:100]
                return ("correction.tool", value)
        
        # 英文模式
        if "wrong" in content_lower or "incorrect" in content_lower:
            # 找"should be"后面的内容
            if "should be" in content_lower:
                idx = content_lower.index("should be") + len("should be")
                value = content[idx:].strip()[:100]
                return ("correction.target", value)
            # 找"use"后面的内容
            if "use" in content_lower:
                idx = content_lower.index("use") + len("use")
                value = content[idx:].strip()[:100]
                return ("correction.tool", value)
        
        return (None, None)
    
    def _extract_preference(self, content: str, content_lower: str) -> tuple:
        """提取偏好信号
        
        模式：
        - "我喜欢XXX" → key=preference.like, value=XXX
        - "以后都用XXX" → key=preference.default, value=XXX
        """
        # 中文模式
        if "我喜欢" in content:
            idx = content.index("我喜欢") + len("我喜欢")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.like", value)
        if "以后都" in content:
            idx = content.index("以后都") + len("以后都")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.default", value)
        if "默认用" in content:
            idx = content.index("默认用") + len("默认用")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.default", value)
        if "偏好" in content:
            idx = content.index("偏好") + len("偏好")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.like", value)
        
        # 英文模式
        if "i prefer" in content_lower:
            idx = content_lower.index("i prefer") + len("i prefer")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.like", value)
        if "always use" in content_lower:
            idx = content_lower.index("always use") + len("always use")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.default", value)
        if "default to" in content_lower:
            idx = content_lower.index("default to") + len("default to")
            value = content[idx:].strip()[:100]
            if value:
                return ("preference.default", value)
        
        return (None, None)
    
    def _extract_decision(self, content: str, content_lower: str) -> tuple:
        """提取决策信号
        
        模式：
        - "决定了，用XXX" → key=decision.choice, value=XXX
        - "就用这个" → key=decision.choice, value=this
        """
        # 中文模式
        if "决定了" in content:
            idx = content.index("决定了") + len("决定了")
            value = content[idx:].strip()[:100]
            if value:
                return ("decision.choice", value)
        if "就用这个" in content:
            return ("decision.choice", "this")
        if "采用" in content:
            idx = content.index("采用") + len("采用")
            value = content[idx:].strip()[:100]
            return ("decision.choice", value)
        
        # 英文模式
        if "decided" in content_lower:
            idx = content_lower.index("decided") + len("decided")
            value = content[idx:].strip()[:100]
            if value:
                return ("decision.choice", value)
        if "go with" in content_lower:
            idx = content_lower.index("go with") + len("go with")
            value = content[idx:].strip()[:100]
            return ("decision.choice", value)
        
        return (None, None)
    
    def _extract_pattern(self, content: str, content_lower: str) -> tuple:
        """提取模式信号
        
        模式：
        - "每次都XXX" → key=pattern.recurring, value=XXX
        - "老是XXX" → key=pattern.recurring, value=XXX
        """
        # 中文模式
        if "每次都" in content:
            idx = content.index("每次都") + len("每次都")
            value = content[idx:].strip()[:100]
            return ("pattern.recurring", value)
        if "老是" in content:
            idx = content.index("老是") + len("老是")
            value = content[idx:].strip()[:100]
            return ("pattern.recurring", value)
        if "经常" in content:
            idx = content.index("经常") + len("经常")
            value = content[idx:].strip()[:100]
            return ("pattern.recurring", value)
        
        # 英文模式
        if "every time" in content_lower:
            idx = content_lower.index("every time") + len("every time")
            value = content[idx:].strip()[:100]
            return ("pattern.recurring", value)
        if "always" in content_lower:
            idx = content_lower.index("always") + len("always")
            value = content[idx:].strip()[:100]
            return ("pattern.recurring", value)
        
        return (None, None)
    
    def _calculate_confidence(self, content: str, pattern_type: str) -> float:
        """计算信号置信度
        
        基于：
        1. 模式类型的权重
        2. 内容长度（越长越具体，置信度越高）
        3. 关键词密度（关键词越多，置信度越高）
        """
        # 基础权重
        base_weight = self.patterns.weight(pattern_type)
        
        # 内容长度因子（100字符以上为满分）
        length_factor = min(len(content) / 100, 1.0)
        
        # 关键词密度因子
        pattern = self.patterns.get(pattern_type)
        if pattern:
            keyword_count = sum(1 for kw in pattern.keywords if kw in content.lower())
            keyword_factor = min(keyword_count / 3, 1.0)
        else:
            keyword_factor = 0.5
        
        # 综合置信度
        confidence = base_weight * 0.4 + length_factor * 0.3 + keyword_factor * 0.3
        return min(confidence, 1.0)
    
    def _deduplicate(self, signals: List[Signal]) -> List[Signal]:
        """去重：相同key+相似value的信号只保留一个"""
        seen: Dict[str, Signal] = {}
        
        for signal in signals:
            # 用key+value前50字符作为去重键
            dedup_key = f"{signal.type}:{signal.key}:{str(signal.value)[:50]}"
            
            if dedup_key not in seen:
                seen[dedup_key] = signal
            else:
                # 保留置信度更高的
                if signal.confidence > seen[dedup_key].confidence:
                    seen[dedup_key] = signal
        
        return list(seen.values())
    
    def close(self) -> None:
        """关闭数据库连接"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
