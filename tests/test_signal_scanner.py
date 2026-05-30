"""测试SessionScanner和SignalIngester"""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from openllm_memory.capsule import (
    Capsule,
    SessionScanner,
    Signal,
    SignalIngester,
    SignalIngesterWithFallback,
    SignalPatterns,
)


@pytest.fixture
def temp_db():
    """创建临时测试数据库"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # 创建messages表和FTS5索引
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, tokenize='trigram')
    """)
    
    # 插入测试数据
    test_messages = [
        # 纠正信号
        (1, "session_001", "user", "不对，应该是Python而不是Java", time.time() - 3600),
        (2, "session_001", "assistant", "好的，我明白了", time.time() - 3500),
        
        # 偏好信号
        (3, "session_002", "user", "我喜欢简洁的代码风格", time.time() - 7200),
        (4, "session_002", "assistant", "明白，我会保持简洁", time.time() - 7100),
        
        # 决策信号
        (5, "session_003", "user", "决定了，用pytest做测试", time.time() - 10800),
        (6, "session_003", "assistant", "好的，使用pytest", time.time() - 10700),
        
        # 模式信号
        (7, "session_004", "user", "每次都忘了导入os模块", time.time() - 14400),
        (8, "session_004", "assistant", "我会记住的", time.time() - 14300),
        
        # 英文信号
        (9, "session_005", "user", "I prefer using type hints", time.time() - 18000),
        (10, "session_005", "assistant", "Got it, I'll use type hints", time.time() - 17900),
        
        # 普通消息（不应被提取）
        (11, "session_006", "user", "今天天气怎么样？", time.time() - 21600),
        (12, "session_006", "assistant", "今天天气不错", time.time() - 21500),
    ]
    
    conn.executemany(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        test_messages,
    )
    
    # 同步到FTS5索引
    conn.execute("INSERT INTO messages_fts(rowid, content) SELECT id, content FROM messages")
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # 清理
    os.unlink(db_path)


@pytest.fixture
def temp_capsule():
    """创建临时胶囊"""
    with tempfile.TemporaryDirectory() as tmpdir:
        capsule = Capsule.open(tmpdir)
        yield capsule


class TestSignalPatterns:
    """测试信号模式库"""
    
    def test_all_patterns(self):
        patterns = SignalPatterns()
        all_patterns = patterns.all()
        
        assert len(all_patterns) == 4
        types = [p[0] for p in all_patterns]
        assert "correction" in types
        assert "preference" in types
        assert "decision" in types
        assert "pattern" in types
    
    def test_keywords(self):
        patterns = SignalPatterns()
        
        # 检查中文关键词
        correction = patterns.get("correction")
        assert correction is not None
        assert "不对" in correction.keywords
        assert "错了" in correction.keywords
        
        # 检查英文关键词
        assert "wrong" in correction.keywords
        assert "incorrect" in correction.keywords
    
    def test_weights(self):
        patterns = SignalPatterns()
        
        assert patterns.weight("correction") == 1.0
        assert patterns.weight("preference") == 0.8
        assert patterns.weight("decision") == 0.7
        assert patterns.weight("pattern") == 0.5
    
    def test_add_custom(self):
        patterns = SignalPatterns()
        patterns.add_custom("custom", ["custom_keyword"], weight=0.6)
        
        custom = patterns.get("custom")
        assert custom is not None
        assert "custom_keyword" in custom.keywords
        assert custom.weight == 0.6


class TestSessionScanner:
    """测试Session Scanner"""
    
    def test_scan_basic(self, temp_db):
        scanner = SessionScanner(temp_db)
        signals = scanner.scan(days=1)
        
        # 应该找到4个信号（纠正、偏好、决策、模式）
        assert len(signals) >= 4
        
        # 检查信号类型
        signal_types = {s.type for s in signals}
        assert "correction" in signal_types
        assert "preference" in signal_types
        assert "decision" in signal_types
        assert "pattern" in signal_types
        
        scanner.close()
    
    def test_scan_english(self, temp_db):
        scanner = SessionScanner(temp_db)
        signals = scanner.scan(days=1)
        
        # 应该找到英文偏好信号
        english_signals = [s for s in signals if "type hints" in str(s.value)]
        assert len(english_signals) >= 1
        
        scanner.close()
    
    def test_scan_empty(self, temp_db):
        scanner = SessionScanner(temp_db)
        
        # 扫描未来的时间，应该没有信号
        signals = scanner.scan(days=-1)
        assert len(signals) == 0
        
        scanner.close()
    
    def test_scan_with_context_manager(self, temp_db):
        with SessionScanner(temp_db) as scanner:
            signals = scanner.scan(days=1)
            assert len(signals) >= 4
    
    def test_signal_confidence(self, temp_db):
        scanner = SessionScanner(temp_db)
        signals = scanner.scan(days=1)
        
        # 所有信号的置信度应该在0-1之间
        for signal in signals:
            assert 0.0 <= signal.confidence <= 1.0
        
        scanner.close()


class TestSignalIngester:
    """测试信号入库管道"""
    
    def test_ingest_basic(self, temp_db, temp_capsule):
        scanner = SessionScanner(temp_db)
        ingester = SignalIngester(temp_capsule)
        
        signals = scanner.scan(days=1)
        written = ingester.ingest(signals)
        
        # 应该写入了部分信号
        assert written > 0
        
        # 检查统计信息
        stats = ingester.get_stats()
        assert stats["total"] == len(signals)
        assert stats["written"] == written
        
        scanner.close()
    
    def test_ingest_dedup(self, temp_db, temp_capsule):
        scanner = SessionScanner(temp_db)
        ingester = SignalIngester(temp_capsule)
        
        # 第一次入库
        signals = scanner.scan(days=1)
        written1 = ingester.ingest(signals)
        
        # 第二次入库（应该去重）
        written2 = ingester.ingest(signals)
        
        # 第二次应该写入更少（或0）
        assert written2 <= written1
        
        scanner.close()
    
    def test_ingest_min_confidence(self, temp_db, temp_capsule):
        scanner = SessionScanner(temp_db)
        ingester = SignalIngester(temp_capsule)
        
        signals = scanner.scan(days=1)
        
        # 用高置信度阈值，应该写入更少
        written_high = ingester.ingest(signals, min_confidence=0.9)
        
        # 用低置信度阈值，应该写入更多
        written_low = ingester.ingest(signals, min_confidence=0.1)
        
        assert written_low >= written_high
        
        scanner.close()


class TestSignalIngesterWithFallback:
    """测试带回滚的信号入库管道"""
    
    def test_fallback_basic(self, temp_db, temp_capsule):
        scanner = SessionScanner(temp_db)
        ingester = SignalIngesterWithFallback(temp_capsule, use_similarity_gate=False)
        
        signals = scanner.scan(days=1)
        written = ingester.ingest(signals)
        
        assert written > 0
        
        stats = ingester.get_stats()
        assert stats["total"] == len(signals)
        assert stats["written"] == written
        
        scanner.close()
    
    def test_fallback_dedup(self, temp_db, temp_capsule):
        scanner = SessionScanner(temp_db)
        ingester = SignalIngesterWithFallback(temp_capsule, use_similarity_gate=False)
        
        # 第一次入库
        signals = scanner.scan(days=1)
        written1 = ingester.ingest(signals)
        
        # 第二次入库（应该去重）
        written2 = ingester.ingest(signals)
        
        assert written2 <= written1
        
        scanner.close()


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self, temp_db, temp_capsule):
        """测试完整流程：扫描 → 入库 → 检索"""
        # 1. 扫描信号
        with SessionScanner(temp_db) as scanner:
            signals = scanner.scan(days=1)
            assert len(signals) >= 4
        
        # 2. 入库信号
        ingester = SignalIngester(temp_capsule)
        written = ingester.ingest(signals)
        assert written > 0
        
        # 3. 检索信号
        state = temp_capsule.state()
        
        # 检查是否有signal/开头的key
        signal_keys = [k for k in state.keys() if k.startswith("signal/")]
        assert len(signal_keys) > 0
        
        # 检查信号内容
        for key in signal_keys:
            value = state[key]
            assert "value" in value
            assert "confidence" in value
            assert "source_session" in value
    
    def test_capsule_prefetch(self, temp_db, temp_capsule):
        """测试用prefetch检索信号"""
        # 入库信号
        with SessionScanner(temp_db) as scanner:
            signals = scanner.scan(days=1)
        
        ingester = SignalIngester(temp_capsule)
        ingester.ingest(signals)
        
        # 用prefetch检索（用signal类型作为查询）
        result = temp_capsule.prefetch("correction")
        assert result  # 应该找到纠正信号
        
        result = temp_capsule.prefetch("preference")
        assert result  # 应该找到偏好信号
        
        # 检查状态
        state = temp_capsule.state()
        signal_keys = [k for k in state.keys() if k.startswith("signal/")]
        assert len(signal_keys) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
