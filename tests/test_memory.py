"""openllm-memory 独立测试。"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule, CAPSULE_DIM


class TestTextCapsule:
    def test_create(self):
        tc = TextCapsule(session_id="s1", insights=["测试"])
        assert tc.session_id == "s1"
        assert tc.insights == ["测试"]

    def test_roundtrip(self):
        tc = TextCapsule(session_id="s1", decisions=[{"summary": "决策1"}], insights=["洞察1"], unresolved=["问题1"])
        d = tc.to_dict()
        tc2 = TextCapsule.from_dict(d)
        assert tc2.insights == ["洞察1"]
        assert tc2.unresolved == ["问题1"]


class TestDeltaCapsule:
    def test_create(self):
        v = np.ones(CAPSULE_DIM, dtype=np.float32)
        dc = DeltaCapsule(session_id="s1", vector=v)
        assert dc.session_id == "s1"
        assert dc.norm > 0

    def test_accumulate(self):
        d1 = DeltaCapsule(session_id="s1", vector=np.ones(CAPSULE_DIM, dtype=np.float32))
        d2 = DeltaCapsule(session_id="s2", vector=np.ones(CAPSULE_DIM, dtype=np.float32) * 2)
        d1.accumulate(d2.vector)
        expected = np.ones(CAPSULE_DIM) * 3
        assert np.allclose(d1.vector, expected, atol=0.01)

    def test_empty_vector(self):
        dc = DeltaCapsule(session_id="s1", vector=np.zeros(CAPSULE_DIM))
        assert dc.norm == 0.0

    def test_roundtrip(self):
        dc = DeltaCapsule(session_id="s1", vector=np.random.randn(CAPSULE_DIM).astype(np.float32) * 0.1)
        d = dc.to_dict()
        dc2 = DeltaCapsule.from_dict(d)
        assert np.allclose(dc.vector, dc2.vector, atol=1e-5)


class TestMemoryOS:
    def test_write_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            tc = TextCapsule(session_id="s1", insights=["测试"])
            path = mos.write(tc)
            assert Path(path).exists()

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            tc = TextCapsule(session_id="s1", decisions=[{"summary": "测试决策"}], insights=["测试洞察"])
            dc = DeltaCapsule(session_id="s1", vector=np.random.randn(CAPSULE_DIM).astype(np.float32) * 0.01)
            mos.write(tc, dc)
            ctx = mos.read()
            assert ctx["status"] == "restored"
            assert "测试决策" in str(ctx["decisions"])
            assert "测试洞察" in ctx["insights"]

    def test_empty_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            ctx = mos.read()
            assert ctx["status"] == "empty"

    def test_checkpoint_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            for i in range(6):
                tc = TextCapsule(session_id=f"s{i}", insights=[f"洞察{i}"])
                dc = DeltaCapsule(session_id=f"s{i}", vector=np.ones(CAPSULE_DIM, dtype=np.float32) * 0.05)
                mos.write(tc, dc)
            cp_files = list(Path(tmp).glob("checkpoint_*.json"))
            assert len(cp_files) >= 1

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            tc = TextCapsule(session_id="s1")
            mos.write(tc)
            s = mos.stats
            assert s["capsules_v06"] == 1

    def test_cross_session_remember(self):
        """核心验证：跨会话记忆恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            # 会话1
            mos1 = MemoryOS(tmp)
            tc = TextCapsule(session_id="s1", decisions=[{"summary": "确认OpenLLM方向"}], insights=["六维Agent躯体"])
            dc = DeltaCapsule(session_id="s1", vector=np.random.randn(CAPSULE_DIM).astype(np.float32) * 0.05)
            mos1.write(tc, dc)

            # 会话2（新实例）
            mos2 = MemoryOS(tmp)
            ctx = mos2.read()
            assert "确认OpenLLM方向" in str(ctx["decisions"])
            assert "六维Agent躯体" in ctx["insights"]

    def test_delta_accumulation_across_sessions(self):
        """Δ跨会话累积。"""
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            # 3次会话，每次小Δ
            for i in range(3):
                tc = TextCapsule(session_id=f"s{i}")
                dc = DeltaCapsule(session_id=f"s{i}", vector=np.ones(CAPSULE_DIM, dtype=np.float32) * 0.03)
                mos.write(tc, dc)
            assert mos.delta is not None
            assert mos.delta.norm > 0.08  # 3次累加
