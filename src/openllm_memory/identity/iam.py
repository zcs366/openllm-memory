"""Iam — 动态自我叙事

如果说SOUL是"我是谁"的固定声明，
Iam就是"我认为我是谁"的动态演化。

它是一个由Agent自己书写的、随时间演进的自我描述。
每次深层反思后，Agent更新Iam——就像人随着年龄成长
对自己有新的认识。

Iam的内容不是固定的，但它记录了自己的变化历史。
所以你可以看到"这个Agent三个月前觉得自己是什么样，
现在又觉得自己是什么样"。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Iam:
    """自我叙事——Agent对自己的动态理解
    
    Iam是Agent可以自己修改的自我描述。
    每次修改都会记录在变更日志中，形成"自我认识的演化史"。
    
    典型Iam内容：
    {
        "self_description": "我是一个专注记忆系统的AI助手...",
        "strengths": ["deep analysis", "memory systems", "writing"],
        "weaknesses": ["over-thinking", "impatience"],
        "current_focus": "building openllm-memory",
        "relationship_with_user": "老搭档关系，信任度高"
    }
    """
    
    def __init__(self, store_dir: str, capsule):
        self._dir = Path(store_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._capsule = capsule
        self._data: Dict[str, Any] = {}
        self._history: List[Dict] = []
        self._load()
    
    def _load(self) -> None:
        """加载Iam状态"""
        history_file = self._dir / "iam_history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text())
                self._data = data.get("current", {})
                self._history = data.get("history", [])
            except (json.JSONDecodeError, OSError):
                pass
    
    def update(self, new_data: Dict[str, Any], 
               reason: str = "",
               session_id: str = "") -> None:
        """更新自我叙事
        
        Args:
            new_data: 新的自我描述数据（会与现有数据合并）
            reason: 更新的原因
            session_id: 来源会话
        """
        # 记录历史
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous": dict(self._data),
            "delta": dict(new_data),
            "reason": reason,
            "session_id": session_id,
        }
        self._history.append(snapshot)
        
        # 应用更新
        self._data.update(new_data)
        
        # 同时写入Δ胶囊（跨会话持久化）
        self._capsule.write("_iam", self._data, op="set",
                           session_id=session_id,
                           metadata={"reason": reason})
        
        # 持久化
        self._save()
    
    def _save(self) -> None:
        history_file = self._dir / "iam_history.json"
        payload = {
            "current": self._data,
            "history": self._history[-100:],  # 最多保留100条历史
        }
        history_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def all(self) -> Dict[str, Any]:
        return dict(self._data)
    
    def history(self, n: int = 10) -> List[Dict]:
        """获取最近的n次自我叙事变更"""
        return list(reversed(self._history[-n:]))
    
    def __repr__(self) -> str:
        return f"Iam({len(self._history)} updates)"
