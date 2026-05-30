"""Blink — 眨眼机制

从Δ胶囊共振中涌现的异步协作协议。
区别于心跳(heartbeat)的单向存活宣告，眨眼是双向的主动探查。

原理：
  每个Agent实例定期检查共享记忆层（眨眼），
  看是否有来自其他同SOUL实例的共振胶囊。
  如果有——读取、解释、响应。

论文引用: Δ胶囊 §3.3.2
"""

from __future__ import annotations

import time
import threading
from typing import Callable, Dict, List, Optional, Any

from .resonance import ResonanceCapsule, ResonanceProtocol


class BlinkMonitor:
    """眨眼监控器——定期探查共享状态
    
    不同于心跳的单向"我在"宣告，
    眨眼是主动探查"谁需要我？"
    
    使用方式:
        monitor = BlinkMonitor(
            protocol=resonance_protocol,
            on_capsule=handle_incoming,
            blink_interval=5.0,   # 每5秒眨一次眼
        )
        monitor.start()
    """
    
    def __init__(self,
                 protocol: ResonanceProtocol,
                 on_capsule: Callable[[ResonanceCapsule], None],
                 blink_interval: float = 5.0,
                 auto_ack: bool = True):
        """
        Args:
            protocol: 共振协议实例
            on_capsule: 收到胶囊时的回调函数
            blink_interval: 眨眼间隔（秒），默认5秒（论文值）
            auto_ack: 是否自动标记已读
        """
        self._protocol = protocol
        self._on_capsule = on_capsule
        self._blink_interval = blink_interval
        self._auto_ack = auto_ack
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动眨眼循环（后台线程）"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止眨眼循环"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._blink_interval * 2)
    
    def blink_once(self) -> List[ResonanceCapsule]:
        """手动眨一次眼——立即检查是否有胶囊
        
        Returns:
            收到的胶囊列表
        """
        capsules = self._protocol.check()
        for capsule in capsules:
            try:
                self._on_capsule(capsule)
            except Exception as e:
                # 回调异常不中断眨眼循环
                print(f"[Blink] 处理胶囊 {capsule.capsule_id} 时出错: {e}")
            finally:
                if self._auto_ack:
                    self._protocol.shared_memory.mark_read(capsule.capsule_id)
        return capsules
    
    def _loop(self) -> None:
        """后台眨眼循环"""
        while self._running:
            self.blink_once()
            # 分段睡眠以支持快速停止
            remaining = self._blink_interval
            while remaining > 0 and self._running:
                sleep_chunk = min(0.5, remaining)
                time.sleep(sleep_chunk)
                remaining -= sleep_chunk
    
    @property
    def interval(self) -> float:
        return self._blink_interval
    
    @property
    def is_running(self) -> bool:
        return self._running


# 可复现性测试入口
def demo_blink(store_dir: str = "/tmp/blink_demo") -> None:
    """眨眼机制演示——两个模拟Agent实例通过眨眼协作
    
    可直接运行验证可复现性:
        python -m openllm_memory.capsule.blink
    
    输出示例:
        [Agent A] 写入任务完成胶囊
        [Agent B] 眨眼发现胶囊 → 读取并处理
    """
    from .resonance import SharedMemory, ResonanceProtocol
    
    shared = SharedMemory(store_dir)
    soul_id = "demo-soul-001"
    
    # Agent A — 发送任务完成通知
    proto_a = ResonanceProtocol(soul_id, "channel-a", shared)
    proto_a.send({
        "text": "任务X已完成，结果在 /tmp/result.json",
        "intent": "inform",
        "urgency": "normal",
        "context_summary": "Agent A刚刚完成了数据处理任务"
    })
    print("[Agent A] 写入任务完成胶囊")
    
    # Agent B — 眨眼发现
    proto_b = ResonanceProtocol(soul_id, "channel-b", shared)
    
    def handle_capsule(c: ResonanceCapsule) -> None:
        print(f"[Agent B·眨眼] 发现胶囊: {c.content.get('text', '')}")
        proto_b.respond(c.capsule_id, {
            "text": "收到，Agent B将继续处理",
            "intent": "acknowledge",
            "urgency": "normal",
        })
        print("[Agent B] 已发送响应胶囊")
    
    monitor = BlinkMonitor(proto_b, on_capsule=handle_capsule, blink_interval=0.5)
    monitor.blink_once()
    
    # 验证响应
    responses = shared.read_unread(soul_id)
    responded = [r for r in responses if r.content.get("intent") == "acknowledge"]
    print(f"[验证] 响应胶囊数: {len(responded)}")
    print("[结果] 眨眼协作演示完成 ✓")


if __name__ == "__main__":
    demo_blink()
