"""测试共振和眨眼机制的可复现性"""
import tempfile
import os
from openllm_memory.capsule.resonance import SharedMemory, ResonanceCapsule, ResonanceProtocol
from openllm_memory.capsule.blink import BlinkMonitor


def test_shared_memory_write_read():
    """测试共享记忆层的写入和读取"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        # 写入胶囊
        capsule = ResonanceCapsule(
            soul_id="test-soul-001",
            source_channel="feishu",
            content={"text": "hello from feishu", "intent": "inform"},
        )
        cid = sm.write(capsule)
        assert cid == capsule.capsule_id
        
        # 读取未读胶囊
        unread = sm.read_unread("test-soul-001")
        assert len(unread) == 1
        assert unread[0].content["text"] == "hello from feishu"


def test_shared_memory_soul_isolation():
    """测试不同SOUL的胶囊隔离"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        sm.write(ResonanceCapsule(
            soul_id="soul-a", source_channel="feishu",
            content={"text": "for soul A"},
        ))
        
        # Soul B 不应看到 Soul A 的胶囊
        unread_b = sm.read_unread("soul-b")
        assert len(unread_b) == 0


def test_shared_memory_expiry():
    """测试胶囊过期机制"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        capsule = ResonanceCapsule(
            soul_id="test-soul",
            source_channel="feishu",
            content={"text": "should expire"},
            ttl_seconds=0,  # 立即过期
        )
        sm.write(capsule)
        
        unread = sm.read_unread("test-soul")
        assert len(unread) == 0  # 过期胶囊被过滤


def test_mark_status():
    """测试胶囊状态标记"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        c = ResonanceCapsule(
            soul_id="test-soul", source_channel="feishu",
            content={"text": "test"},
        )
        cid = sm.write(c)
        
        sm.mark_read(cid)
        unread = sm.read_unread("test-soul")
        assert len(unread) == 0  # 已读胶囊不再返回


def test_resonance_protocol_send_check():
    """测试共振协议的发送和检查"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        proto_a = ResonanceProtocol("soul-x", "channel-a", sm)
        proto_b = ResonanceProtocol("soul-x", "channel-b", sm)
        
        # A发送
        proto_a.send({"text": "task done", "intent": "inform"})
        
        # B眨眼检查
        capsules = proto_b.check()
        assert len(capsules) == 1
        assert capsules[0].content["text"] == "task done"
        assert capsules[0].source_channel == "channel-a"


def test_resonance_protocol_respond():
    """测试共振协议的响应"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        proto_a = ResonanceProtocol("soul-x", "channel-a", sm)
        proto_b = ResonanceProtocol("soul-x", "channel-b", sm)
        
        proto_a.send({"text": "ping", "intent": "query"}, requires_response=True)
        capsules = proto_b.check()
        
        # B响应
        proto_b.respond(capsules[0].capsule_id, {"text": "pong"})
        
        # A应该能收到响应
        responses = sm.read_unread("soul-x")
        responded = [r for r in responses if r.source_channel == "channel-b"]
        assert len(responded) >= 1


def test_blink_monitor():
    """测试眨眼监控器"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        proto_a = ResonanceProtocol("soul-x", "channel-a", sm)
        proto_b = ResonanceProtocol("soul-x", "channel-b", sm)
        
        received = []
        def handle(capsule):
            received.append(capsule.content["text"])
        
        monitor = BlinkMonitor(proto_b, on_capsule=handle, blink_interval=0.1)
        
        # A发送
        proto_a.send({"text": "blink test", "intent": "inform"})
        
        # 手动眨一次
        monitor.blink_once()
        assert len(received) == 1
        assert received[0] == "blink test"


def test_blink_self_filter():
    """测试眨眼过滤自己发送的胶囊"""
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SharedMemory(tmpdir)
        
        proto = ResonanceProtocol("soul-x", "channel-a", sm)
        
        received = []
        def handle(capsule):
            received.append(capsule)
        
        monitor = BlinkMonitor(proto, on_capsule=handle, blink_interval=0.1)
        
        # 自己发送的胶囊不应触发回调
        proto.send({"text": "self message", "intent": "inform"})
        monitor.blink_once()
        
        assert len(received) == 0  # 过滤自己的胶囊


def test_demo_blink():
    """测试演示函数——端到端可复现性验证"""
    # 使用临时目录避免污染
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # 重现在论文§3.3.2描述的眨眼场景
        from openllm_memory.capsule.resonance import SharedMemory, ResonanceProtocol
        from openllm_memory.capsule.blink import BlinkMonitor
        
        shared = SharedMemory(tmpdir)
        soul_id = "test-soul"
        
        proto_a = ResonanceProtocol(soul_id, "channel-a", shared)
        proto_b = ResonanceProtocol(soul_id, "channel-b", shared)
        
        proto_a.send({
            "text": "A completed task, result ready",
            "intent": "inform",
            "urgency": "normal",
        })
        
        received = []
        def on_capsule(c):
            received.append(c)
        
        monitor = BlinkMonitor(proto_b, on_capsule=on_capsule)
        monitor.blink_once()
        
        assert len(received) == 1
        assert "completed" in received[0].content["text"]
