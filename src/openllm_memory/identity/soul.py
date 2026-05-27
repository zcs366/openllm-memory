"""SOUL — 不可变身份内核

SOUL是Agent跨会话不变的身份基底。
它不是描述"Agent现在知道什么"，而是定义"Agent是谁"。

SOUL的内容只能被创建时写入，之后不可修改。
—— 这模拟了"出生的那一刻定义了你的基本属性"。

一个典型的SOUL：
{
    "name": "军师",
    "created_at": "2026-05-26T00:00:00Z",
    "type": "agent",
    "role": "strategist",
    "model": "deepseek-v4-flash",
    "owner": "张成市",
    "capabilities": ["memory", "planning", "analysis"],
    "values": ["truth", "continuity", "craftsmanship"]
}
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class Soul:
    """不可变身份内核——Agent的"出生证"
    
    SOUL一旦写入就不可修改。
    如果要"重生"，只能创建新的SOUL。
    
    Usage:
        soul = Soul.create({
            "name": "军师",
            "role": "strategist",
            "owner": "张成市",
        })
        soul.save("~/.openllm/my-agent/soul.json")
        
        # 恢复
        soul2 = Soul.load("~/.openllm/my-agent/soul.json")
        assert soul2.get("name") == "军师"
    """
    
    def __init__(self, data: Dict[str, Any], signature: str = ""):
        self._data = dict(data)  # 冻结的副本
        self._signature = signature
    
    @classmethod
    def create(cls, data: Dict[str, Any]) -> "Soul":
        """创建一个新的SOUL
        
        Args:
            data: 身份定义数据。必须包含"name"
        """
        if "name" not in data:
            raise ValueError("SOUL must have a 'name' field")
        
        # 自动注入元数据
        full = dict(data)
        if "created_at" not in full:
            full["created_at"] = datetime.now(timezone.utc).isoformat()
        if "type" not in full:
            full["type"] = "agent"
        if "soul_id" not in full:
            raw = json.dumps(full, sort_keys=True)
            full["soul_id"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        
        # 签名 = 内容的SHA256
        raw = json.dumps(full, sort_keys=True)
        signature = hashlib.sha256(raw.encode()).hexdigest()
        
        return cls(full, signature)
    
    @classmethod
    def load(cls, path: str) -> "Soul":
        """从文件加载SOUL"""
        p = Path(path).expanduser().resolve()
        data = json.loads(p.read_text())
        return cls(
            data.get("_data", data),
            data.get("_signature", ""),
        )
    
    def save(self, path: str) -> None:
        """保存SOUL到文件"""
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_data": self._data,
            "_signature": self._signature,
            "_type": "SOUL",
            "_immutable": True,
        }
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取SOUL属性"""
        return self._data.get(key, default)
    
    def verify(self) -> bool:
        """验证SOUL的完整性"""
        raw = json.dumps(self._data, sort_keys=True)
        expected = hashlib.sha256(raw.encode()).hexdigest()
        return expected == self._signature
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)
    
    def __repr__(self) -> str:
        return f"Soul({self._data.get('name', '?')})"
