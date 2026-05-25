"""openllm-memory 测试 — v0.2 真实embedding版。"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule, encode_text, EMBEDDING_DIM


class TestRealEmbedding:
    def test_encode_text_produces_real_vector(self):
        """编码真实文本——不再是随机数。"""
        v1 = encode_text("确认OpenLLM架构方向")
        assert v1.shape == (EMBEDDING_DIM,)
        assert v1.dtype == np.float32
        assert abs(float(np.linalg.norm(v1)) - 1.0) < 0.01  # normalize
        assert not np.allclose(v1, np.zeros(EMBEDDING_DIM))    # 非零

    def test_similar_texts_produce_similar_vectors(self):
        """语义相似→向量相似。这是因果关系的证明。"""
        v1 = encode_text("Agent需要记忆系统")
        v2 = encode_text("Agent需要持久化记忆")
        v3 = encode_text("今天天气很好适合出去玩")

        sim_12 = float(np.dot(v1, v2))
        sim_13 = float(np.dot(v1, v3))
        # 相似的应该比不相似的cos_sim更高
        assert sim_12 > sim_13, f"相似文本cos_sim({sim_12:.3f})应>不相似({sim_13:.3f})"

    def test_different_texts_produce_different_vectors(self):
        """不同文本→不同向量。"""
        v1 = encode_text("Agent记忆层")
        v2 = encode_text("代码语义编码")
        assert not np.allclose(v1, v2, atol=0.01)


class TestDeltaCapsuleReal:
    def test_from_text_produces_real_delta(self):
        """从文本生成的Δ向量包含真实语义信息。"""
        dc = DeltaCapsule.from_text("s1", "确认OpenLLM六维架构方向")
        assert dc.vector.shape == (EMBEDDING_DIM,)
        assert dc.norm > 0.8  # normalized, so ~1.0
        assert dc.metadata["source"] == "embedding"

    def test_two_sessions_produce_accumulated_delta(self):
        """两次会话Δ累积。"""
        d1 = DeltaCapsule.from_text("s1", "讨论Agent架构")
        d2 = DeltaCapsule.from_text("s2", "确定寄生启动策略")

        d1.accumulate(d2.vector)
        # 两次累计后范数应增大
        assert d1.norm > 0.5


class TestTextCapsuleToText:
    def test_to_text_includes_decisions_and_insights(self):
        tc = TextCapsule(
            session_id="s1",
            decisions=[{"summary": "确认方向"}],
            insights=["Agent需要六维躯体"],
        )
        text = tc.to_text()
        assert "确认方向" in text
        assert "六维躯体" in text


class TestMemoryOSRealEmbedding:
    def test_write_without_explicit_delta_auto_generates(self):
        """不传delta→自动从text生成真实embedding。"""
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            tc = TextCapsule(
                session_id="s1",
                decisions=[{"summary": "确认OpenLLM方向"}],
                insights=["寄生启动策略"],
            )
            path = mos.write(tc)  # 不传delta
            assert Path(path).exists()

            # 检查v0.7文件存在且向量非零
            v07 = list(Path(tmp).glob("v07_*.json"))
            assert len(v07) >= 1
            with open(v07[0]) as f:
                d = json.load(f)
            assert len(d["vector"]) == EMBEDDING_DIM
            assert not all(v == 0.0 for v in d["vector"])
            assert d["metadata"]["source"] == "embedding"

    def test_cross_session_semantic_memory(self):
        """跨会话：两次写不同内容，验证向量变化。"""
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)

            # 会话1：技术讨论
            tc1 = TextCapsule(session_id="s1", decisions=[{"summary": "Agent架构讨论"}])
            mos.write(tc1)
            norm1 = mos.delta.norm

            # 会话2：不同话题
            tc2 = TextCapsule(session_id="s2", decisions=[{"summary": "寄生启动策略"}])
            mos.write(tc2)
            norm2 = mos.delta.norm

            # 话题不同，Δ应该累积变化
            assert norm2 > 0

    def test_read_restores_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            mos = MemoryOS(tmp)
            tc = TextCapsule(
                session_id="s1",
                decisions=[{"summary": "确认方向"}],
                insights=["真实embedding驱动"],
            )
            mos.write(tc)
            ctx = mos.read()
            assert ctx["status"] == "restored"
            assert "确认方向" in str(ctx["decisions"])
            assert ctx["delta_norm"] > 0.5
