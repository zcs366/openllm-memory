"""Δ 操作类型定义"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import hashlib


@dataclass
class Delta:
    """一个Δ操作——记忆的最小不可分单位
    
    每个Δ有唯一ID、类型、数据、时间戳。
    这是记忆系统的"原子"——不可拆分，不可篡改。
    """
    op: str                          # 操作类型: set/delete/merge/append
    key: str                         # 记忆键
    value: Any                       # 值
    timestamp: float = 0.0           # 时间戳 (epoch seconds)
    delta_id: str = ""               # 唯一ID (SHA1 of content)
    session_id: str = ""             # 来源会话
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).timestamp()
        if not self.delta_id:
            self.delta_id = self._compute_id()
    
    def _compute_id(self) -> str:
        raw = f"{self.op}|{self.key}|{json.dumps(self.value, sort_keys=True)}|{self.timestamp}|{self.session_id}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "op": self.op,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Delta":
        return cls(
            op=d["op"],
            key=d["key"],
            value=d.get("value"),
            timestamp=d.get("timestamp", 0.0),
            delta_id=d.get("delta_id", ""),
            session_id=d.get("session_id", ""),
            metadata=d.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        return f"Δ({self.op} {self.key} @{self.timestamp:.1f})"


class DeltaOp:
    """Δ操作类型常量"""
    SET = "set"         # 设置/覆盖值
    DELETE = "delete"   # 删除键
    MERGE = "merge"     # 合并到现有值（dict merge）
    APPEND = "append"   # 追加到数组
    PATCH = "patch"     # 部分更新（JSON patch风格）
