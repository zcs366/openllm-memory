"""信号模式库——定义4种信号类型的关键词模式

信号类型：
1. correction - 用户纠正Agent的错误
2. preference - 用户表达偏好
3. decision - 用户做出决策
4. pattern - 用户识别到模式

每种类型有中英文关键词，用于FTS5搜索。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SignalPattern:
    """单个信号模式"""
    type: str  # correction/preference/decision/pattern
    keywords: List[str]  # 关键词列表
    weight: float = 1.0  # 权重（用于置信度计算）
    description: str = ""  # 描述


class SignalPatterns:
    """信号模式库
    
    用法：
        patterns = SignalPatterns()
        for ptype, keywords in patterns.all():
            for keyword in keywords:
                # 搜索该关键词
                pass
    """
    
    # 纠正模式（最高优先级）
    CORRECTION_KEYWORDS = [
        # 中文
        "不对", "错了", "应该是", "别这样", "不要", "停止",
        "不是这个", "搞错了", "重新来", "重来",
        # 英文
        "wrong", "incorrect", "no,", "stop", "don't",
        "not right", "that's not", "I said", "I meant",
        "actually", "correction", "fix this",
    ]
    
    # 偏好模式（次高优先级）
    PREFERENCE_KEYWORDS = [
        # 中文
        "我喜欢", "以后都", "默认用", "总是", "永远不",
        "偏好", "习惯", "风格", "方式",
        # 英文
        "I prefer", "always use", "never use", "default to",
        "I like", "I don't like", "going forward", "from now on",
        "remember that", "keep in mind", "make sure to",
    ]
    
    # 决策模式（中等优先级）
    DECISION_KEYWORDS = [
        # 中文
        "决定了", "就用这个", "选A", "选B", "采用",
        "确认", "批准", "同意", "通过",
        # 英文
        "decided", "go with", "use this", "switch to",
        "let's go with", "I decided", "we're using", "the plan is",
        "chosen", "picked", "decision", "we agreed",
    ]
    
    # 模式模式（较低优先级）
    PATTERN_KEYWORDS = [
        # 中文
        "每次都", "老是", "经常", "总是", "又",
        "重复", "循环", "习惯性", "典型",
        # 英文
        "again", "every time", "keep", "always",
        "as usual", "same as before", "like last time",
        "we always", "the usual", "recurring",
    ]
    
    def __init__(self):
        self._patterns: Dict[str, SignalPattern] = {
            "correction": SignalPattern(
                type="correction",
                keywords=self.CORRECTION_KEYWORDS,
                weight=1.0,  # 最高权重
                description="用户纠正Agent的错误",
            ),
            "preference": SignalPattern(
                type="preference",
                keywords=self.PREFERENCE_KEYWORDS,
                weight=0.8,
                description="用户表达偏好",
            ),
            "decision": SignalPattern(
                type="decision",
                keywords=self.DECISION_KEYWORDS,
                weight=0.7,
                description="用户做出决策",
            ),
            "pattern": SignalPattern(
                type="pattern",
                keywords=self.PATTERN_KEYWORDS,
                weight=0.5,
                description="用户识别到模式",
            ),
        }
    
    def get(self, type: str) -> Optional[SignalPattern]:
        """获取指定类型的模式"""
        return self._patterns.get(type)
    
    def all(self) -> List[Tuple[str, List[str]]]:
        """获取所有模式（类型, 关键词列表）"""
        return [
            (p.type, p.keywords)
            for p in self._patterns.values()
        ]
    
    def all_keywords(self) -> List[Tuple[str, str]]:
        """获取所有关键词（类型, 关键词）"""
        result = []
        for p in self._patterns.values():
            for keyword in p.keywords:
                result.append((p.type, keyword))
        return result
    
    def weight(self, type: str) -> float:
        """获取指定类型的权重"""
        p = self._patterns.get(type)
        return p.weight if p else 0.0
    
    def types(self) -> List[str]:
        """获取所有类型"""
        return list(self._patterns.keys())
    
    def add_custom(self, type: str, keywords: List[str], 
                   weight: float = 0.5, description: str = "") -> None:
        """添加自定义模式"""
        if type not in self._patterns:
            self._patterns[type] = SignalPattern(
                type=type,
                keywords=[],
                weight=weight,
                description=description,
            )
        self._patterns[type].keywords.extend(keywords)
