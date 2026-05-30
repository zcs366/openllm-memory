"""Resonance — 跨频道共振协议

共享记忆层实现。同一SOUL的不同频道实例通过写入/读取
共振胶囊实现状态同步——不是消息传递，是共享状态解释。

论文引用: Δ胶囊 §3.3
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ResonanceCapsule:
    """共振胶囊——跨频道通信的基本单元
    
    一个实例写入，其他实例以自身模型权重独立解释。
    """
    soul_id: str
    source_channel: str          # 如 "feishu", "yuanbao", "telegram"
    content: Dict[str, Any]      # {text, intent, urgency, context_summary}
    capsule_id: str = ""
    target_channel: Optional[str] = None  # None = 广播给所有同SOUL实例
    requires_response: bool = False
    ttl_seconds: int = 3600
    timestamp: float = 0.0
    status: str = "pending"      # pending | read | responded | expired
    
    def __post_init__(self):
        if not self.capsule_id:
            self.capsule_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).timestamp()
    
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "soul_id": self.soul_id,
            "source_channel": self.source_channel,
            "target_channel": self.target_channel,
            "content": self.content,
            "requires_response": self.requires_response,
            "ttl_seconds": self.ttl_seconds,
            "timestamp": self.timestamp,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResonanceCapsule":
        return cls(
            soul_id=d["soul_id"],
            source_channel=d["source_channel"],
            content=d["content"],
            capsule_id=d.get("capsule_id", ""),
            target_channel=d.get("target_channel"),
            requires_response=d.get("requires_response", False),
            ttl_seconds=d.get("ttl_seconds", 3600),
            timestamp=d.get("timestamp", 0.0),
            status=d.get("status", "pending"),
        )


class SharedMemory:
    """共享记忆层——跨频道共振的物理载体
    
    使用本地JSON文件作为共享状态存储。
    所有共享同一SOUL的实例通过此层交换胶囊。
    
    生产环境可替换为Redis/etcd/分布式文件系统。
    """
    
    def __init__(self, store_dir: str):
        self._dir = Path(store_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._capsules_file = self._dir / "resonance_capsules.json"
    
    def _load_all(self) -> List[Dict]:
        if not self._capsules_file.exists():
            return []
        try:
            return json.loads(self._capsules_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    
    def _save_all(self, capsules: List[Dict]) -> None:
        self._capsules_file.write_text(
            json.dumps(capsules, indent=2, ensure_ascii=False)
        )
    
    def write(self, capsule: ResonanceCapsule) -> str:
        """写入共振胶囊到共享层
        
        Returns:
            capsule_id: 胶囊唯一标识
        """
        capsules = self._load_all()
        capsules.append(capsule.to_dict())
        self._save_all(capsules)
        return capsule.capsule_id
    
    def read_unread(self, soul_id: str, channel: Optional[str] = None) -> List[ResonanceCapsule]:
        """读取该SOUL的未读胶囊
        
        返回未过期、未读、匹配SOUL的胶囊。
        自动过滤已过期胶囊。
        """
        capsules = self._load_all()
        unread = []
        now = time.time()
        modified = False
        
        for cap_dict in capsules:
            capsule = ResonanceCapsule.from_dict(cap_dict)
            
            # 跳过不匹配SOUL的
            if capsule.soul_id != soul_id:
                continue
            
            # 跳过已过期
            if capsule.is_expired():
                cap_dict["status"] = "expired"
                modified = True
                continue
            
            # 跳过已读/已响应/已过期
            if capsule.status in ("read", "responded", "expired"):
                continue
            
            # 频道过滤（如果指定了target_channel）
            if capsule.target_channel and channel and capsule.target_channel != channel:
                continue
            
            unread.append(capsule)
        
        if modified:
            self._save_all(capsules)
        
        return unread
    
    def mark_read(self, capsule_id: str) -> None:
        self._update_status(capsule_id, "read")
    
    def mark_responded(self, capsule_id: str) -> None:
        self._update_status(capsule_id, "responded")
    
    def mark_expired(self, capsule_id: str) -> None:
        self._update_status(capsule_id, "expired")
    
    def _update_status(self, capsule_id: str, status: str) -> None:
        capsules = self._load_all()
        for cap_dict in capsules:
            if cap_dict.get("capsule_id") == capsule_id:
                cap_dict["status"] = status
                break
        self._save_all(capsules)
    
    def cleanup_expired(self) -> int:
        """清理所有过期胶囊，返回清理数量"""
        capsules = self._load_all()
        count = len(capsules)
        capsules = [c for c in capsules 
                    if not ResonanceCapsule.from_dict(c).is_expired()]
        self._save_all(capsules)
        return count - len(capsules)


class ResonanceProtocol:
    """共振协议——论文算法1的完整实现
    
    每个Agent实例持有此对象，在事件循环中调用 resonance_loop()。
    """
    
    def __init__(self, 
                 soul_id: str,
                 channel: str,
                 shared_memory: SharedMemory,
                 poll_interval: float = 5.0):
        self.soul_id = soul_id
        self.channel = channel
        self.shared_memory = shared_memory
        self.poll_interval = poll_interval
    
    def send(self, 
             content: Dict[str, Any],
             target_channel: Optional[str] = None,
             requires_response: bool = False,
             ttl_seconds: int = 3600) -> str:
        """发送共振胶囊
        
        Args:
            content: {text, intent, urgency, context_summary}
            target_channel: None=广播, 指定=定向发送
            requires_response: 是否需要响应胶囊
            ttl_seconds: 过期时间
        
        Returns:
            capsule_id
        """
        capsule = ResonanceCapsule(
            soul_id=self.soul_id,
            source_channel=self.channel,
            content=content,
            target_channel=target_channel,
            requires_response=requires_response,
            ttl_seconds=ttl_seconds,
        )
        return self.shared_memory.write(capsule)
    
    def check(self) -> List[ResonanceCapsule]:
        """检查是否有来自其他实例的胶囊（眨眼动作）"""
        capsules = self.shared_memory.read_unread(self.soul_id, self.channel)
        results = []
        for capsule in capsules:
            # 跳过自己发的
            if capsule.source_channel == self.channel:
                self.shared_memory.mark_read(capsule.capsule_id)
                continue
            
            results.append(capsule)
        return results
    
    def respond(self, 
                to_capsule_id: str,
                content: Dict[str, Any]) -> str:
        """响应一个共振胶囊"""
        self.shared_memory.mark_responded(to_capsule_id)
        content.setdefault("intent", "acknowledge")
        return self.send(content, requires_response=False)
    
    def get_poll_interval(self) -> float:
        return self.poll_interval
