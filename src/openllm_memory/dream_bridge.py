"""
DreamBridge — Δ胶囊消费ISA Brain梦境日志的桥梁。

从 ISA Brain 的 brain_dream.jsonl 中读取dream事件，
转换为 Δ胶囊 TextCapsule + DeltaCapsule（真实语义向量），
存入 MemoryOS，提供跨Agent的认知查询。
同时将事件注入Arbiter进行认知矛盾检测。

三角架构:
  ISA(管道) → brain_dream.jsonl → DreamBridge → Δ胶囊(共享记忆)
                                       ↓
                                  Arbiter(认知仲裁)
                                                   ↓
                                          查询接口: 谁梦到过什么?
"""

import json
import time
import os
import sys

# HF被墙——强制离线模式，从缓存加载embedding模型
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from pathlib import Path
from datetime import datetime
from typing import Optional

from .core import MemoryOS, TextCapsule, DeltaCapsule, encode_text
import numpy as np


DEFAULT_BRAIN_ROOT = Path.home() / ".hermes" / "isa" / "brain"
DEFAULT_POSITION_FILE = Path.home() / ".openllm" / "dream_bridge_positions.json"


class DreamBridge:
    """
    Δ胶囊之桥: ISA Brain dreaming → Δ胶囊记忆。

    每调用一次 consume()，扫描所有Agent的 brain_dream.jsonl，
    读取新行 → 转为胶囊 → 存入 MemoryOS。
    """

    def __init__(
        self,
        brain_root: Path = DEFAULT_BRAIN_ROOT,
        capsule_dir: str = "~/.openllm/capsules",
        position_file: Path = DEFAULT_POSITION_FILE,
    ):
        self.brain_root = Path(brain_root)
        self.capsule_dir = capsule_dir
        self.position_file = Path(position_file)
        self.position_file.parent.mkdir(parents=True, exist_ok=True)
        self._positions: dict[str, int] = self._load_positions()

    # ── 位置追踪 ────────────────────────────────────────

    def _load_positions(self) -> dict[str, int]:
        """加载每个Agent已读取的行数位置。"""
        if self.position_file.exists():
            try:
                with open(self.position_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_positions(self):
        """持久化位置信息。"""
        with open(self.position_file, "w") as f:
            json.dump(self._positions, f, indent=2)

    def reset_position(self, agent_id: Optional[str] = None):
        """重置位置（用于重新消费）。"""
        if agent_id:
            self._positions.pop(agent_id, None)
        else:
            self._positions.clear()
        self._save_positions()

    # ── 发现Agent ───────────────────────────────────────

    def list_agents(self) -> list[str]:
        """列出brain目录下所有有brain_dream.jsonl的Agent。"""
        agents = []
        if not self.brain_root.exists():
            return agents
        for entry in self.brain_root.iterdir():
            if entry.is_dir():
                dream_log = entry / "brain_dream.jsonl"
                if dream_log.exists():
                    agents.append(entry.name)
        return sorted(agents)

    # ── 消费 ────────────────────────────────────────────

    def consume(self, agent_id: Optional[str] = None) -> dict:
        """
        消费新dream事件 → 存入MemoryOS。

        Args:
            agent_id: 指定Agent，None=全部。

        Returns:
            {agent_id: {"new_events": N, "capsules_written": N, ...}}
        """
        results = {}

        if agent_id:
            agents = [agent_id]
        else:
            agents = self.list_agents()

        for aid in agents:
            dream_log = self.brain_root / aid / "brain_dream.jsonl"
            if not dream_log.exists():
                results[aid] = {"error": "no brain_dream.jsonl", "new_events": 0}
                continue

            # 读取新行
            last_pos = self._positions.get(aid, 0)
            new_events = []
            with open(dream_log, "r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if event.get("type") == "dream":
                                new_events.append(event)
                        except json.JSONDecodeError:
                            continue
                self._positions[aid] = f.tell()

            if not new_events:
                results[aid] = {"new_events": 0, "capsules_written": 0}
                continue

            # 转为胶囊 → 存入MemoryOS + 注入Arbiter
            arbiter_insights = []
            mos = MemoryOS(capsule_dir=str(self.capsule_dir))
            capsules_written = 0
            for event in new_events:
                temperature = event.get("temperature", "neutral")
                for discovery in event.get("discoveries", []):
                    # 构建洞察文本
                    insight_text = (
                        f"Dream: Agent[{aid}] 发现卡片 "
                        f"「{discovery['card_a']}」与「{discovery['card_b']}」的关联 — "
                        f"共享关键词: {', '.join(discovery.get('shared_keywords', []))}"
                    )
                    temperature = event.get("temperature", "neutral")

                    # 创建TextCapsule
                    session_id = f"dream-{aid}-{event['timestamp']}"
                    capsule = TextCapsule(
                        session_id=session_id,
                        timestamp=datetime.fromisoformat(event["timestamp"]).timestamp(),
                        insights=[insight_text],
                        decisions=[{
                            "summary": f"[{temperature}] {insight_text}",
                            "agent": aid,
                        }],
                        outputs=[],
                        unresolved=[],
                    )

                    # 用洞察文本生成真实语义向量
                    text_for_embedding = f"{insight_text} temperature:{temperature}"
                    delta_vec = DeltaCapsule.from_text(session_id, text_for_embedding)

                    # 写入MemoryOS
                    mos.write(capsule, delta=delta_vec)
                    capsules_written += 1

                # 上层事件: 记录聚合洞察+收集arbiter输入
                if len(event.get("discoveries", [])) > 1:
                    summary_insight = (
                        f"Dream聚合: Agent[{aid}] 本轮发现 "
                        f"{len(event['discoveries'])} 组卡片关联 "
                        f"(温度: {temperature})"
                    )
                    # 收集insight供Arbiter仲裁
                    for d in event.get("discoveries", []):
                        arbiter_insights.append({
                            "agent_id": aid,
                            "card_id": d["card_a"],
                            "content": f"发现{d['card_a']}与{d['card_b']}的关联: {', '.join(d.get('shared_keywords', []))}",
                            "timestamp": event["timestamp"],
                        })
                    agg_capsule = TextCapsule(
                        session_id=f"dream-agg-{aid}-{event['timestamp']}",
                        timestamp=datetime.fromisoformat(event["timestamp"]).timestamp(),
                        insights=[summary_insight],
                        decisions=[{"summary": f"[{temperature}] {summary_insight}", "agent": aid}],
                    )
                    agg_delta = DeltaCapsule.from_text(agg_capsule.session_id, f"{summary_insight} temperature:{temperature}")
                    mos.write(agg_capsule, delta=agg_delta)
                    capsules_written += 1

            # 注入Arbiter——检测认知矛盾
            conflicts_detected = 0
            if arbiter_insights:
                try:
                    sys.path.insert(0, str(Path.home() / "projects" / "isa"))
                    from arbiter import Arbiter
                    arbiter = Arbiter()
                    conflicts = arbiter.detect_conflicts(arbiter_insights)
                    for c in conflicts:
                        ruling = arbiter.arbitrate(c)
                        if ruling["verdict"] in ("contradiction", "stalemate"):
                            conflicts_detected += 1
                            # 将矛盾也记入胶囊
                            conflict_capsule = TextCapsule(
                                session_id=f"arbitration-{aid}-{datetime.now().isoformat()}",
                                insights=[f"[Arbiter] ⚔️ {ruling['resolution']}"],
                                decisions=[{"summary": ruling["resolution"], "agent": "arbiter"}],
                            )
                            mos.write(conflict_capsule)
                            capsules_written += 1
                except ImportError:
                    pass  # Arbiter不可用时静默继续

            self._save_positions()
            results[aid] = {
                "new_events": len(new_events),
                "capsules_written": capsules_written,
                "discoveries": sum(len(e.get("discoveries", [])) for e in new_events),
                "conflicts_detected": conflicts_detected,
            }

        return {
            "agents_processed": len(results),
            "details": results,
        }

    # ── 查询 ────────────────────────────────────────────

    def query(self, keyword: str, limit: int = 10) -> list[dict]:
        """
        查询谁梦到过什么。关键词匹配insight文本。

        未来升级为语义搜索（用DeltaCapsule向量做cosine sim）。
        """
        mos = MemoryOS(capsule_dir=str(self.capsule_dir))
        results = []
        for v06_file in sorted(Path(self.capsule_dir).expanduser().glob("v06_dream-*.json"), reverse=True):
            try:
                with open(v06_file, "r") as f:
                    data = json.load(f)
                insights = data.get("insights", [])
                for ins in insights:
                    if keyword.lower() in ins.lower():
                        results.append({
                            "session_id": data.get("session_id"),
                            "timestamp": data.get("timestamp"),
                            "insight": ins,
                            "decisions": data.get("decisions", []),
                        })
                        break
            except (json.JSONDecodeError, OSError):
                continue
            if len(results) >= limit:
                break
        return results

    def hot_topics(self, min_occurrences: int = 2) -> list[dict]:
        """
        最热门的卡片关联对。
        统计所有dream事件中 (card_a ↔ card_b) 的出现次数。
        """
        pair_counts: dict[tuple[str, str], int] = {}
        for agent_dir in self.brain_root.iterdir():
            if not agent_dir.is_dir():
                continue
            dream_log = agent_dir / "brain_dream.jsonl"
            if not dream_log.exists():
                continue
            with open(dream_log, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        if event.get("type") != "dream":
                            continue
                        for d in event.get("discoveries", []):
                            a, b = sorted([d["card_a"], d["card_b"]])
                            key = (a, b)
                            pair_counts[key] = pair_counts.get(key, 0) + 1
                    except (json.JSONDecodeError, KeyError):
                        continue

        hot = [
            {"card_a": a, "card_b": b, "occurrences": n}
            for (a, b), n in sorted(pair_counts.items(), key=lambda x: -x[1])
            if n >= min_occurrences
        ]
        return hot


def cli():
    """命令行入口: python3 -m openllm_memory.dream_bridge [consume|query|hot]"""
    import sys

    bridge = DreamBridge()

    if len(sys.argv) < 2:
        print("用法: python3 -m openllm_memory.dream_bridge [consume|query <keyword>|hot|agents|reset]")
        return

    cmd = sys.argv[1]

    if cmd == "consume":
        agent = sys.argv[2] if len(sys.argv) > 2 else ""
        result = bridge.consume(agent if agent else None)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif cmd == "query":
        if len(sys.argv) < 3:
            print("需提供关键词: query <keyword>")
            return
        results = bridge.query(sys.argv[2])
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif cmd == "hot":
        results = bridge.hot_topics()
        if results:
            print("热门卡片关联:")
            for r in results:
                print(f"  {r['card_a']} ↔ {r['card_b']}  ({r['occurrences']}次)")
        else:
            print("暂无热门关联")

    elif cmd == "agents":
        agents = bridge.list_agents()
        print(f"找到 {len(agents)} 个Agent:")
        for a in agents:
            print(f"  - {a}")

    elif cmd == "reset":
        agent = sys.argv[2] if len(sys.argv) > 2 else None
        bridge.reset_position(agent)
        print(f"已重置{'全部' if agent is None else f'Agent[{agent}]'}位置")

    else:
        print(f"未知命令: {cmd}")


if __name__ == "__main__":
    cli()
