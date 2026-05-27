"""仲裁层——冲突解决协议

当两个Agent会话同时写了同一份记忆，发生了分歧，
仲裁层决定谁赢，或者如何合并。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .delta import Delta


@dataclass
class Conflict:
    """一个冲突记录"""
    key: str
    session_a_delta: Delta
    session_b_delta: Delta
    resolved: bool = False
    resolution: Optional[str] = None  # "a_wins" | "b_wins" | "merged"
    resolved_value: Any = None


class Arbitrator:
    """仲裁器——决定冲突如何解决
    
    策略（从强到弱）：
    1. explicit: 显式标记的优先级（session级别）
    2. timestamp: 时间戳较新的胜出
    3. merge_append: 双方都保留，标记来源
    4. merge_deep: 逐字段合并（对dict值有效）
    """
    
    def __init__(self, strategy: str = "timestamp"):
        """Args:
            strategy: 仲裁策略，默认timestamp
                      可选: explicit / timestamp / merge_append / merge_deep
        """
        self.strategy = strategy
        self._session_priority: Dict[str, int] = {}
    
    def set_priority(self, session_id: str, priority: int) -> None:
        """设置会话的仲裁优先级（数字越大越优先）"""
        self._session_priority[session_id] = priority
    
    def resolve(self, conflict: Conflict) -> Any:
        """解决单个冲突"""
        if conflict.resolved:
            return conflict.resolved_value
        
        a, b = conflict.session_a_delta, conflict.session_b_delta
        resolved_value, resolution = self._apply_strategy(a, b, conflict.key)
        
        conflict.resolved_value = resolved_value
        conflict.resolution = resolution
        conflict.resolved = True
        return resolved_value
    
    def resolve_batch(self, diffs: List[Dict]) -> Dict[str, Any]:
        """批量解决多个键的冲突
        
        Args:
            diffs: reconcile()输出的差异列表
        
        Returns:
            合并后的状态
        """
        result = {}
        for diff in diffs:
            key = diff["key"]
            ours, theirs = diff["ours"], diff["theirs"]
            
            # 根据策略选择值
            if self.strategy == "explicit":
                value = ours  # 默认自己优先
            elif self.strategy == "merge_deep":
                if isinstance(ours, dict) and isinstance(theirs, dict):
                    merged = theirs.copy()
                    merged.update(ours)  # ours覆盖theirs
                    value = merged
                else:
                    value = ours
            else:
                value = ours
            
            keys = key.split(".")
            self._deep_set(result, keys, value)
        
        return result
    
    def _apply_strategy(self, a: Delta, b: Delta, key: str
                        ) -> tuple[Any, str]:
        """应用仲裁策略"""
        
        # 1. explicit — 会话优先级
        if self.strategy == "explicit":
            a_prio = self._session_priority.get(a.session_id, 0)
            b_prio = self._session_priority.get(b.session_id, 0)
            if a_prio > b_prio:
                return a.value, "a_wins"
            elif b_prio > a_prio:
                return b.value, "b_wins"
        
        # 2. timestamp — 新的覆盖旧的
        if self.strategy == "timestamp":
            if a.timestamp >= b.timestamp:
                return a.value, "a_wins"
            else:
                return b.value, "b_wins"
        
        # 3. merge_append — 双方都保留
        if self.strategy == "merge_append":
            merged = {
                "_a": a.value,
                "_b": b.value,
                "_merged_at": max(a.timestamp, b.timestamp),
            }
            return merged, "merged"
        
        # 4. merge_deep
        if self.strategy == "merge_deep":
            if isinstance(a.value, dict) and isinstance(b.value, dict):
                merged = b.value.copy()
                merged.update(a.value)
                return merged, "merged"
            return a.value, "a_wins"
        
        # 默认：a赢
        return a.value, "a_wins"
    
    @staticmethod
    def _deep_set(d: Dict, keys: List[str], value: Any) -> None:
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
