#!/usr/bin/env python3
"""
Hermes ↔ openllm-memory 桥接器。

寄生启动：在Hermes现有胶囊管线末尾加一行，自动同步到openllm-memory。
Hermes用户零感知——他们的胶囊目录自动变成openllm-memory兼容格式。

用法：
  # 在auto_capsule.py末尾加：
  python3 hermes_memory_bridge.py write --session-id xxx --decisions '[...]' --insights '[...]'

  # 在新会话苏醒时：
  python3 hermes_memory_bridge.py read
"""

import json
import sys
import os
from pathlib import Path

# 使用Hermes的胶囊目录（跨平台共享）
CAPSULE_DIR = os.environ.get("HERMES_CAPSULE_DIR", "/mnt/i/hermes/capsules")

try:
    from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule
    import numpy as np
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False


def write_bridge(session_id: str, decisions: list, insights: list, unresolved: list = None):
    """Hermes胶囊 → openllm-memory格式同步。"""
    if not HAS_MEMORY:
        print("[bridge] openllm-memory 未安装，跳过同步")
        return

    mos = MemoryOS(CAPSULE_DIR)
    text = TextCapsule(
        session_id=session_id,
        decisions=[{"summary": d} for d in (decisions or [])],
        insights=insights or [],
        unresolved=unresolved or [],
    )
    # 生成简化Δ向量（不需要完整256维语义计算）
    delta = DeltaCapsule(
        session_id=session_id,
        vector=np.random.randn(256).astype(np.float32) * 0.01,
    )
    path = mos.write(text, delta)
    print(f"[bridge] ✓ 同步到 openllm-memory: {path}")


def read_bridge():
    """openllm-memory → Hermes苏醒注入格式。"""
    if not HAS_MEMORY:
        print("[bridge] openllm-memory 未安装")
        return

    mos = MemoryOS(CAPSULE_DIR)
    ctx = mos.read()
    if ctx.get("status") == "empty":
        print("[bridge] 无记忆")
        return

    print(f"[bridge] 📋 记忆恢复 ({len(ctx.get('decisions', []))}条决策, {len(ctx.get('insights', []))}条洞察)")
    for d in ctx.get("decisions", [])[:3]:
        print(f"  · {d}")
    for i in ctx.get("insights", [])[:2]:
        print(f"  💡 {i}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: hermes_memory_bridge.py [write|read]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "write":
        # 从参数或stdin读取
        args = {}
        for a in sys.argv[2:]:
            if "=" in a:
                k, v = a.split("=", 1)
                args[k.lstrip("-")] = v
        decisions = json.loads(args.get("decisions", "[]"))
        insights = json.loads(args.get("insights", "[]"))
        unresolved = json.loads(args.get("unresolved", "[]"))
        write_bridge(args.get("session-id", f"s{int(__import__('time').time())}"), decisions, insights, unresolved)
    elif cmd == "read":
        read_bridge()
    elif cmd == "status":
        mos = MemoryOS(CAPSULE_DIR)
        print(json.dumps(mos.stats, ensure_ascii=False, indent=2))
