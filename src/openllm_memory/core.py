"""
openllm-memory — Δ胶囊记忆层。真实语义embedding驱动。

寄生启动策略：不是另一个Agent框架——是每个Agent框架最好的记忆层。
pip install openllm-memory → 你的Agent从此记得用户。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

# ── embedding模型配置 ─────────────────────────────
# 默认384维（all-MiniLM-L6-v2），零GPU依赖，CPU跑
EMBEDDING_DIM = 384
CHECKPOINT_INTERVAL = 5
DELTA_NORM_THRESHOLD = 0.3

# 懒加载embedding模型（首次使用时才下载）
_embedder = None
_embedder_name = "all-MiniLM-L6-v2"


def _get_embedder():
    """懒加载embedding模型。首次调用时下载。"""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(_embedder_name)
        except ImportError:
            return None
    return _embedder


def encode_text(text: str) -> np.ndarray:
    """将任意文本编码为语义向量。384维FP32。"""
    model = _get_embedder()
    if model is None:
        # fallback: 无sentence-transformers时用hash-based模拟
        return _fallback_encode(text)
    return model.encode(text, normalize_embeddings=True).astype(np.float32)


def _fallback_encode(text: str) -> np.ndarray:
    """无embedding模型时的降级方案：基于字符hash的确定性向量。"""
    rng = np.random.RandomState(hash(text) % (2**31))
    return rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.1


@dataclass
class TextCapsule:
    """v0.6 可读胶囊。给人看，可审计。"""
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

    def to_text(self) -> str:
        """将胶囊转为可被embedding编码的文本。"""
        parts = []
        for d in self.decisions:
            parts.append(d.get("summary", str(d)))
        for i in self.insights:
            parts.append(i)
        return " | ".join(parts) if parts else self.session_id

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
    """
    v0.7 Δ语义向量胶囊。

    从真实对话文本提取embedding，计算两次会话的语义位移。
    不再是随机数——向量与对话内容有因果关系。
    """
    session_id: str
    vector: np.ndarray       # EMBEDDING_DIM维FP32语义向量
    norm: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.vector.shape != (EMBEDDING_DIM,):
            self.vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        self.norm = float(np.linalg.norm(self.vector))

    @classmethod
    def from_text(cls, session_id: str, text: str) -> "DeltaCapsule":
        """从文本生成Δ向量——真实的语义编码。"""
        vec = encode_text(text)
        return cls(session_id=session_id, vector=vec, metadata={"source": "embedding", "model": _embedder_name})

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
            vector=np.array(d.get("vector", [0.0] * EMBEDDING_DIM), dtype=np.float32),
            timestamp=d.get("timestamp", time.time()),
            metadata=d.get("metadata", {}),
        )


class MemoryOS:
    """记忆操作系统。"""

    def __init__(self, capsule_dir: str = "~/.openllm/capsules"):
        self.capsule_dir = Path(capsule_dir).expanduser()
        self.capsule_dir.mkdir(parents=True, exist_ok=True)
        self.session_count = 0
        self.delta: Optional[DeltaCapsule] = None
        self.text: Optional[TextCapsule] = None

    def write(self, text: TextCapsule, delta: Optional[DeltaCapsule] = None) -> str:
        """写入胶囊。如果delta为None，从text自动生成语义Δ向量。"""
        self.text = text
        self.session_count += 1

        v06_path = self.capsule_dir / f"v06_{text.session_id}.json"
        with open(v06_path, "w", encoding="utf-8") as f:
            json.dump(text.to_dict(), f, ensure_ascii=False, indent=2)

        # 如果没有显式传入delta，从text自动生成真实语义向量
        if delta is None:
            delta = DeltaCapsule.from_text(text.session_id, text.to_text())

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
            "delta_norm": round(self.delta.norm, 4) if self.delta else 0.0,
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
            "delta_norm": round(self.delta.norm, 4) if self.delta else 0.0,
        }
