"""
openllm-memory — Δ Capsule 记忆系统

Agent身份连续性的基础设施。
记忆不是查出来的——它是Agent活着的方式。

Usage:
    from openllm_memory import Capsule
    capsule = Capsule.open("~/.openllm/my-agent")
    capsule.write("note", {"content": "something to remember"})
"""

from .capsule.core import Capsule
from .capsule.delta import Delta, DeltaOp
from .identity.soul import Soul
from .identity.iam import Iam
from .capsule.resonance import SharedMemory, ResonanceCapsule, ResonanceProtocol
from .capsule.blink import BlinkMonitor

__all__ = [
    "Capsule",
    "Delta",
    "DeltaOp",
    "Soul",
    "Iam",
    # 共振与眨眼
    "SharedMemory",
    "ResonanceCapsule",
    "ResonanceProtocol",
    "BlinkMonitor",
]
__version__ = "1.1.1"
