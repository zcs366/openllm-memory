"""检查点——序列化与恢复

胶囊的"快照"能力。关机时保存全部Δ的聚合状态，
开机时从检查点恢复，无需重放全部历史。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Checkpoint:
    """检查点管理器
    
    快照 = 当前全部状态 + Δ计数 + 元数据。
    加载时优先用检查点，再增量追加快于检查点的Δ。
    """
    
    def __init__(self, checkpoints_dir: str):
        self._dir = Path(checkpoints_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, state: Dict[str, Any], delta_count: int,
             metadata: Dict[str, Any] = None) -> str:
        """保存检查点
        
        Returns:
            检查点文件名
        """
        cp = {
            "state": state,
            "delta_count": delta_count,
            "timestamp": time.time(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        filename = f"cp-{int(time.time())}.json"
        (self._dir / filename).write_text(
            json.dumps(cp, indent=2, ensure_ascii=False)
        )
        return filename
    
    def load_latest(self) -> Optional[Dict]:
        """加载最新的检查点
        
        Returns:
            检查点数据，或None（无检查点）
        """
        files = sorted(self._dir.glob("cp-*.json"), reverse=True)
        if not files:
            return None
        try:
            return json.loads(files[0].read_text())
        except (json.JSONDecodeError, OSError):
            return None
    
    def list(self) -> List[Dict]:
        """列出所有检查点"""
        result = []
        for f in sorted(self._dir.glob("cp-*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                result.append({
                    "file": f.name,
                    "delta_count": data.get("delta_count", 0),
                    "timestamp": data.get("timestamp", 0),
                    "created_at": data.get("created_at", ""),
                    "size": f.stat().st_size,
                })
            except (json.JSONDecodeError, OSError):
                result.append({"file": f.name, "error": "corrupt"})
        return result
    
    def clean_old(self, keep: int = 5) -> int:
        """清理旧检查点，保留最近keep个
        
        Returns:
            删除的文件数
        """
        files = sorted(self._dir.glob("cp-*.json"), reverse=True)
        if len(files) <= keep:
            return 0
        removed = 0
        for f in files[keep:]:
            f.unlink()
            removed += 1
        return removed
