"""
openllm-memory — Δ胶囊记忆层。零依赖。任何Agent框架可挂载。

寄生启动策略：不是另一个Agent框架——是每个Agent框架最好的记忆层。
pip install openllm-memory → 你的Agent从此记得用户。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


CAPSULE_DIM = 256
CHECKPOINT_INTERVAL = 5
DELTA_NORM_THRESHOLD = 0.3


@dataclass
class TextCapsule:
    """v0.6 可读胶囊。"""
    session_id: str
    timestamp: float = field(default_factory=time.time)
    decisions: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "timestamp": self.timestamp,
            "decisions": self.decisions, "outputs": self.outputs,
            "insights": self.insights, "unresolved": self.unresolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TextCapsule":
        return cls(
            session_id=d.get("session_id", ""),
            timestamp=d.get("timestamp", time.time()),
            decisions=d.get("decisions", []),
            insights=d.get("insights", []),
            unresolved=d.get("unresolved", []),
        )


@dataclass
class DeltaCapsule:
    """v0.7 Δ语义向量胶囊。256维FP32。"""
    session_id: str
    vector: np.ndarray
    norm: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.shape != (CAPSULE_DIM,):
            self.vector = np.zeros(CAPSULE_DIM, dtype=np.float32)
        self.norm = float(np.linalg.norm(self.vector))

    def accumulate(self, other: np.ndarray) -> "DeltaCapsule":
        self.vector = self.vector + np.asarray(other, dtype=np.float32)
        self.norm = float(np.linalg.norm(self.vector))
        return self

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id, "vector": self.vector.tolist(),
            "norm": self.norm, "timestamp": self.timestamp, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaCapsule":
        return cls(
            session_id=d.get("session_id", ""),
            vector=np.array(d.get("vector", [0.0] * CAPSULE_DIM), dtype=np.float32),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


class MemoryOS:
    """
    记忆操作系统。

    三个核心操作：
      write(session) — 会话结束时写入Δ胶囊
      read()          — 新会话开始时恢复记忆
      checkpoint()    — 定期全量快照
    """

    def __init__(self, capsule_dir: str = "~/.openllm/capsules"):
        self.capsule_dir = Path(capsule_dir).expanduser()
        self.capsule_dir.mkdir(parents=True, exist_ok=True)
        self.session_count = 0
        self.delta: Optional[DeltaCapsule] = None
        self.text: Optional[TextCapsule] = None

    def write(self, text: TextCapsule, delta: Optional[DeltaCapsule] = None) -> str:
        self.text = text
        self.session_count += 1

        v06_path = self.capsule_dir / f"v06_{text.session_id}.json"
        with open(v06_path, "w", encoding="utf-8") as f:
            json.dump(text.to_dict(), f, ensure_ascii=False, indent=2)

        if delta is not None:
            if self.delta is not None:
                self.delta.accumulate(delta.vector)
            else:
                self.delta = delta
            v07_path = self.capsule_dir / f"v07_{delta.session_id}.json"
            with open(v07_path, "w", encoding="utf-8") as f:
                json.dump(self.delta.to_dict(), f, ensure_ascii=False, indent=2)

        if self.delta and (
            self.session_count >= CHECKPOINT_INTERVAL
            or self.delta.norm > DELTA_NORM_THRESHOLD
        ):
            self._checkpoint()

        return str(v06_path)

    def read(self) -> dict:
        v06_files = sorted(self.capsule_dir.glob("v06_*.json"), reverse=True)
        if not v06_files:
            return {"status": "empty", "context": "无记忆。第一次对话？"}

        with open(v06_files[0], "r", encoding="utf-8") as f:
            self.text = TextCapsule.from_dict(json.load(f))

        v07_files = sorted(self.capsule_dir.glob("v07_*.json"), reverse=True)
        if v07_files:
            with open(v07_files[0], "r", encoding="utf-8") as f:
                self.delta = DeltaCapsule.from_dict(json.load(f))

        cp_files = sorted(self.capsule_dir.glob("checkpoint_*.json"), reverse=True)
        if cp_files and self.delta:
            with open(cp_files[0], "r", encoding="utf-8") as f:
                cp = json.load(f)
                if cp.get("delta"):
                    self.delta.vector = np.array(cp["delta"], dtype=np.float32) + self.delta.vector
                    self.delta.norm = float(np.linalg.norm(self.delta.vector))

        return {
            "status": "restored",
            "decisions": [d.get("summary", "") for d in (self.text.decisions if self.text else [])],
            "insights": self.text.insights if self.text else [],
            "unresolved": self.text.unresolved if self.text else [],
            "delta_norm": self.delta.norm if self.delta else 0.0,
        }

    def _checkpoint(self) -> str:
        if not self.text or self.delta is None:
            return ""
        cp = {
            "seq_number": self.session_count,
            "timestamp": time.time(),
            "text": self.text.to_dict(),
            "delta": self.delta.vector.tolist(),
        }
        cp_path = self.capsule_dir / f"checkpoint_{self.session_count:03d}.json"
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(cp, f, ensure_ascii=False, indent=2)
        return str(cp_path)

    @property
    def stats(self) -> dict:
        v06 = list(self.capsule_dir.glob("v06_*.json"))
        v07 = list(self.capsule_dir.glob("v07_*.json"))
        cp = list(self.capsule_dir.glob("checkpoint_*.json"))
        return {
            "capsules_v06": len(v06),
            "capsules_v07": len(v07),
            "checkpoints": len(cp),
            "delta_norm": self.delta.norm if self.delta else 0.0,
        }
