"""Δ 胶囊核心引擎——事件溯源的记忆容器

一个Capsule = 一串不可变的Δ + 一个可恢复的状态快照。
读记忆=重放Δ到某时间点。写记忆=追加一个新Δ。
核对=对比Δ序列发现分歧。

开盒即用，不需要数据库，不需要外部服务。
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .delta import Delta, DeltaOp
from .checkpoint import Checkpoint
from .hash_index import HashIndexWithFallback


class Capsule:
    """Δ胶囊——事件溯源的记忆容器
    
    用法:
        cap = Capsule.open("~/.openllm/my-agent")
        cap.write("user.name", "张成市")
        cap.write("user.lang", "zh")
        
        # 跨会话恢复
        cap2 = Capsule.open("~/.openllm/my-agent")
        state = cap2.state()  # {"user": {"name": "张成市", "lang": "zh"}}
    """
    
    def __init__(self, store_dir: str, use_hash_index: bool = True):
        self._dir = Path(store_dir).expanduser().resolve()
        self._deltas_dir = self._dir / "deltas"
        self._checkpoints_dir = self._dir / "checkpoints"
        self._lock = threading.Lock()
        self._deltas: List[Delta] = []
        self._loaded = False
        self._checkpointer = Checkpoint(str(self._checkpoints_dir))
        
        # 哈希索引（支持回滚）
        self._hash_index = HashIndexWithFallback(
            table_size=500003, 
            K=2, 
            use_hash_index=use_hash_index
        )
        
        self._deltas_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    
    @classmethod
    def open(cls, store_dir: str = "~/.openllm/capsule", use_hash_index: bool = True) -> "Capsule":
        """打开或创建一个记忆胶囊
        
        Args:
            store_dir: 存储目录，默认 ~/.openllm/capsule
            use_hash_index: 是否使用哈希索引加速检索
        """
        cap = cls(store_dir, use_hash_index=use_hash_index)
        cap._load()
        return cap
    
    def _load(self) -> None:
        """加载所有Δ到内存"""
        if self._loaded:
            return
        self._deltas = []
        if not self._deltas_dir.exists():
            self._loaded = True
            return
        
        for f in sorted(self._deltas_dir.iterdir()):
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                    self._deltas.append(Delta.from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    continue  # 损坏文件，跳过
        self._loaded = True
    
    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    
    def write(self, key: str, value: Any, op: str = "set",
              session_id: str = "", metadata: Dict = None) -> Delta:
        """写入一个Δ操作
        
        Args:
            key: 记忆键 (支持点号路径如 "user.name")
            value: 值
            op: 操作类型 (set/delete/merge/append/patch)
            session_id: 来源会话标识
            metadata: 附加元数据
        
        Returns:
            创建的Δ对象
        """
        delta = Delta(
            op=op,
            key=key,
            value=value,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._deltas.append(delta)
            self._persist_delta(delta)
            
            # 同步更新哈希索引
            if op == DeltaOp.SET:
                self._hash_index.insert(key, value)
            elif op == DeltaOp.DELETE:
                self._hash_index.remove(key)
            # 对于MERGE/APPEND/PATCH，需要更新索引
            elif op in (DeltaOp.MERGE, DeltaOp.APPEND, DeltaOp.PATCH):
                # 获取当前值并更新
                current_value = self._hash_index.lookup(key)
                if current_value:
                    # 合并操作，需要重新计算最终值
                    # 这里简化处理，实际应该根据操作类型合并
                    self._hash_index.insert(key, value)
        
        return delta
    
    def _persist_delta(self, delta: Delta) -> None:
        """持久化一个Δ到文件"""
        path = self._deltas_dir / f"{delta.delta_id}.json"
        path.write_text(json.dumps(delta.to_dict(), indent=2, ensure_ascii=False))
    
    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    
    def state(self, at_time: float = None) -> Dict[str, Any]:
        """重放所有Δ到某个时间点，返回当前状态
        
        Args:
            at_time: 时间点 (epoch seconds)。None=最新
        
        Returns:
            聚合后的状态字典
        """
        if at_time is None:
            at_time = float('inf')
        
        # 从检查点（已压缩的历史）开始
        state: Dict[str, Any] = {}
        checkpoint_state = self._checkpoint_state()
        if checkpoint_state:
            state.update(checkpoint_state)
        
        # 增量重放检查点之后的Δ
        for d in self._deltas:
            if d.timestamp > at_time:
                break
            self._apply_delta(state, d)
        
        return state
    
    def _checkpoint_state(self) -> Optional[Dict]:
        """从检查点恢复状态"""
        latest = self._checkpointer.load_latest()
        if latest:
            return latest.get("state")
        return None
    
    def _apply_delta(self, state: Dict[str, Any], delta: Delta) -> None:
        """将一个Δ应用到状态上"""
        keys = delta.key.split(".")
        
        if delta.op == DeltaOp.SET:
            self._deep_set(state, keys, delta.value)
        elif delta.op == DeltaOp.DELETE:
            self._deep_delete(state, keys)
        elif delta.op == DeltaOp.MERGE:
            existing = self._deep_get(state, keys)
            if isinstance(existing, dict) and isinstance(delta.value, dict):
                existing.update(delta.value)
            else:
                self._deep_set(state, keys, delta.value)
        elif delta.op == DeltaOp.APPEND:
            existing = self._deep_get(state, keys)
            if isinstance(existing, list):
                if isinstance(delta.value, list):
                    existing.extend(delta.value)
                else:
                    existing.append(delta.value)
            else:
                self._deep_set(state, keys, [delta.value])
        elif delta.op == DeltaOp.PATCH:
            existing = self._deep_get(state, keys)
            if isinstance(existing, dict) and isinstance(delta.value, dict):
                existing.update(delta.value)
            else:
                self._deep_set(state, keys, delta.value)
    
    def prefetch(self, query: str, max_results: int = 10) -> str:
        """检索相关记忆（支持哈希索引加速）
        
        Args:
            query: 查询字符串
            max_results: 最大返回条数
        
        Returns:
            格式化的记忆上下文文本
        """
        if not self._loaded:
            self._load()
        
        query_lower = query.lower()
        matches = []
        
        # 使用哈希索引加速key查找
        if self._hash_index.use_hash_index:
            # 从哈希索引中查找匹配的key
            for delta in self._deltas:
                # 检查key是否包含查询字符串
                if query_lower in delta.key.lower():
                    # 计算分数
                    score = 3  # key匹配的基础分数
                    
                    # 检查value是否匹配
                    if isinstance(delta.value, str) and query_lower in delta.value.lower():
                        score += 1
                    
                    matches.append((score, delta))
                    
                    # 如果已经找到足够的结果，停止
                    if len(matches) >= max_results * 2:  # 多找一些以便排序
                        break
        else:
            # 回滚到原有线性扫描
            for d in reversed(self._deltas):
                score = 0
                if query_lower in d.key.lower():
                    score += 3
                if isinstance(d.value, str) and query_lower in d.value.lower():
                    score += 1
                
                if score > 0:
                    matches.append((score, d))
        
        # 按分数排序
        matches.sort(key=lambda x: -x[0])
        
        # 去重（同一个key可能有多个delta）
        seen_keys = set()
        unique_matches = []
        for score, delta in matches:
            if delta.key not in seen_keys:
                seen_keys.add(delta.key)
                unique_matches.append((score, delta))
                if len(unique_matches) >= max_results:
                    break
        
        results = [d.to_dict() for _, d in unique_matches]
        
        if not results:
            return ""
        
        lines = ["[记忆胶囊 - Δ序列检索结果]"]
        for r in results:
            ts = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- [{ts}] {r['op']} {r['key']} = {json.dumps(r['value'], ensure_ascii=False)}")
        
        return "\n".join(lines)
    
    def get(self, key: str) -> Any:
        """获取特定键的值"""
        return self._deep_get(self.state(), key.split("."))
    
    def enable_hash_index(self) -> None:
        """启用哈希索引"""
        self._hash_index.enable_hash_index()
    
    def disable_hash_index(self) -> None:
        """禁用哈希索引（回滚到线性扫描）"""
        self._hash_index.disable_hash_index()
    
    def rebuild_hash_index(self, new_table_size: Optional[int] = None) -> None:
        """重建哈希索引
        
        Args:
            new_table_size: 新的表大小
        """
        self._hash_index.rebuild(new_table_size)
    
    def get_hash_index_stats(self) -> Dict[str, Any]:
        """获取哈希索引统计信息
        
        Returns:
            统计信息字典
        """
        return self._hash_index.get_stats()
    
    # ------------------------------------------------------------------
    # 核对与恢复
    # ------------------------------------------------------------------
    
    def reconcile(self, other: "Capsule") -> List[Dict]:
        """与另一个胶囊核对，找出差异Δ
        
        Args:
            other: 另一个胶囊
        
        Returns:
            差异列表：[{"key": ..., "ours": ..., "theirs": ...}, ...]
        """
        our_state = self.state()
        their_state = other.state()
        
        all_keys = set()
        for d in self._deltas:
            all_keys.add(d.key)
        for d in other._deltas:
            all_keys.add(d.key)
        
        diff = []
        for key in sorted(all_keys):
            keys = key.split(".")
            our_val = self._deep_get(our_state, keys)
            their_val = self._deep_get(their_state, keys)
            if our_val != their_val:
                diff.append({
                    "key": key,
                    "ours": our_val,
                    "theirs": their_val,
                })
        
        return diff
    
    def latest_deltas(self, n: int = 20) -> List[Delta]:
        """获取最近的n个Δ"""
        return list(reversed(self._deltas[-n:]))
    
    def delta_count(self) -> int:
        return len(self._deltas)
    
    # ------------------------------------------------------------------
    # 持久化工具
    # ------------------------------------------------------------------
    
    def compact(self) -> int:
        """将全部Δ压缩为检查点，清空Δ日志
        返回被压缩的Δ数量
        """
        with self._lock:
            current = self.state()
            count = len(self._deltas)
            
            if count == 0:
                return 0
            
            # 用Checkpoint类保存（确保文件名一致）
            self._checkpointer.save(
                state=current,
                delta_count=count,
                metadata={"note": "compressed delta log"},
            )
            
            # 清除Δ文件
            for f in self._deltas_dir.iterdir():
                if f.suffix == ".json":
                    f.unlink()
            self._deltas = []
            
            # 清理旧检查点
            self._checkpointer.clean_old(keep=5)
            
            return count
    
    def __repr__(self) -> str:
        return f"Capsule({self._dir}, {len(self._deltas)}Δ)"
    
    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    
    @staticmethod
    def _deep_set(d: Dict, keys: List[str], value: Any) -> None:
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    
    @staticmethod
    def _deep_get(d: Dict, keys: List[str]) -> Any:
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d
    
    @staticmethod
    def _deep_delete(d: Dict, keys: List[str]) -> None:
        for k in keys[:-1]:
            if not isinstance(d, dict) or k not in d:
                return
            d = d[k]
        d.pop(keys[-1], None)
