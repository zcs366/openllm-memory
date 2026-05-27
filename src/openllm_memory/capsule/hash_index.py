"""纯Python多头哈希索引（无PyTorch依赖）

实现Engram风格的多头哈希索引，用于加速Δ胶囊的key查找。
严格遵循Pro建议：无PyTorch依赖，64位截断，回滚机制支持。

用法：
    from openllm_memory.capsule.hash_index import HashIndex
    
    index = HashIndex(table_size=500003, K=2)
    index.insert("user.name", "张成市")
    results = index.lookup("user.name")
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple


class HashIndex:
    """纯Python多头哈希索引
    
    实现Engram风格的多头哈希索引，用于加速key查找。
    无PyTorch依赖，使用纯Python实现。
    
    Attributes:
        table_size: 哈希表大小（建议使用质数）
        K: 哈希头数量
        tables: 哈希表列表
        hash_functions: 哈希函数列表
    """
    
    def __init__(self, table_size: int = 500003, K: int = 2):
        """初始化哈希索引
        
        Args:
            table_size: 哈希表大小，建议使用质数以减少碰撞
            K: 哈希头数量，多头哈希可以减少碰撞影响
        """
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        if K <= 0:
            raise ValueError("K must be positive")
        
        self.table_size = table_size
        self.K = K
        self.tables: List[Dict[int, Tuple[str, Any]]] = [{} for _ in range(K)]
        self.hash_functions = [self._make_hash(k) for k in range(K)]
        
        # 统计信息
        self._insert_count = 0
        self._collision_count = 0
    
    def _make_hash(self, seed: int):
        """创建64位截断的哈希函数
        
        Args:
            seed: 哈希种子，不同头使用不同种子
            
        Returns:
            哈希函数
        """
        def hash_fn(key: str) -> int:
            """计算key的哈希值
            
            Args:
                key: 要哈希的字符串
                
            Returns:
                64位哈希值
            """
            # 使用SHA256确保一致性，取前8字节作为64位哈希
            key_bytes = str(key).encode('utf-8')
            hash_bytes = hashlib.sha256(key_bytes + str(seed).encode()).digest()
            # 取前8字节，转换为64位整数
            hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
            return hash_int & 0xFFFFFFFFFFFFFFFF  # 64位截断
        
        return hash_fn
    
    def insert(self, key: str, value: Any) -> bool:
        """插入键值对
        
        Args:
            key: 键
            value: 值
            
        Returns:
            是否插入成功（False表示发生碰撞）
        """
        success = True
        
        for k in range(self.K):
            idx = self.hash_functions[k](key) % self.table_size
            
            # 检查碰撞
            if idx in self.tables[k]:
                existing_key, _ = self.tables[k][idx]
                if existing_key != key:
                    self._collision_count += 1
                    success = False
            
            # 插入或更新
            self.tables[k][idx] = (key, value)
        
        self._insert_count += 1
        return success
    
    def lookup(self, key: str) -> List[Any]:
        """查找键对应的值（多头哈希）
        
        Args:
            key: 要查找的键
            
        Returns:
            匹配的值列表（可能为空）
        """
        results = []
        seen_keys = set()
        
        for k in range(self.K):
            idx = self.hash_functions[k](key) % self.table_size
            
            if idx in self.tables[k]:
                stored_key, stored_value = self.tables[k][idx]
                if stored_key == key and stored_key not in seen_keys:
                    results.append(stored_value)
                    seen_keys.add(stored_key)
        
        return results
    
    def remove(self, key: str) -> bool:
        """删除键值对
        
        Args:
            key: 要删除的键
            
        Returns:
            是否删除成功
        """
        success = False
        
        for k in range(self.K):
            idx = self.hash_functions[k](key) % self.table_size
            
            if idx in self.tables[k]:
                stored_key, _ = self.tables[k][idx]
                if stored_key == key:
                    del self.tables[k][idx]
                    success = True
        
        return success
    
    def contains(self, key: str) -> bool:
        """检查键是否存在
        
        Args:
            key: 要检查的键
            
        Returns:
            是否存在
        """
        for k in range(self.K):
            idx = self.hash_functions[k](key) % self.table_size
            
            if idx in self.tables[k]:
                stored_key, _ = self.tables[k][idx]
                if stored_key == key:
                    return True
        
        return False
    
    def size(self) -> int:
        """获取索引中的键值对数量
        
        Returns:
            键值对数量
        """
        # 使用第一个表统计唯一键
        unique_keys = set()
        for idx, (key, _) in self.tables[0].items():
            unique_keys.add(key)
        return len(unique_keys)
    
    def clear(self) -> None:
        """清空索引"""
        self.tables = [{} for _ in range(self.K)]
        self._insert_count = 0
        self._collision_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息
        
        Returns:
            统计信息字典
        """
        unique_keys = set()
        for idx, (key, _) in self.tables[0].items():
            unique_keys.add(key)
        
        return {
            'table_size': self.table_size,
            'K': self.K,
            'unique_keys': len(unique_keys),
            'insert_count': self._insert_count,
            'collision_count': self._collision_count,
            'collision_rate': self._collision_count / max(1, self._insert_count),
            'load_factor': len(unique_keys) / self.table_size,
        }
    
    def rebuild(self, new_table_size: Optional[int] = None) -> None:
        """重建索引（用于优化碰撞率）
        
        Args:
            new_table_size: 新的表大小，None则自动调整
        """
        # 收集所有唯一的键值对（从所有表中）
        all_items = {}
        for k in range(self.K):
            for idx, (key, value) in self.tables[k].items():
                if key not in all_items:
                    all_items[key] = value
        
        # 确定新表大小
        if new_table_size is None:
            # 自动调整：当前大小的2倍，且为质数
            new_table_size = self._next_prime(self.table_size * 2)
        
        # 重建
        self.table_size = new_table_size
        self.tables = [{} for _ in range(self.K)]
        self._insert_count = 0
        self._collision_count = 0
        
        # 重新插入
        for key, value in all_items.items():
            self.insert(key, value)
    
    def _next_prime(self, n: int) -> int:
        """找到大于n的最小质数
        
        Args:
            n: 起始数
            
        Returns:
            大于n的最小质数
        """
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True
        
        while not is_prime(n):
            n += 1
        return n


class HashIndexWithFallback:
    """带回滚机制的哈希索引
    
    支持在哈希索引和线性扫描之间切换，确保回滚能力。
    
    Attributes:
        hash_index: 哈希索引实例
        use_hash_index: 是否使用哈希索引
        fallback_data: 回滚时使用的原始数据
    """
    
    def __init__(self, table_size: int = 500003, K: int = 2, use_hash_index: bool = True):
        """初始化带回滚机制的哈希索引
        
        Args:
            table_size: 哈希表大小
            K: 哈希头数量
            use_hash_index: 是否使用哈希索引
        """
        self.hash_index = HashIndex(table_size, K)
        self.use_hash_index = use_hash_index
        self.fallback_data: Dict[str, Any] = {}
    
    def insert(self, key: str, value: Any) -> bool:
        """插入键值对
        
        Args:
            key: 键
            value: 值
            
        Returns:
            是否插入成功
        """
        # 总是保存到回滚数据
        self.fallback_data[key] = value
        
        # 如果使用哈希索引，也插入到哈希索引
        if self.use_hash_index:
            return self.hash_index.insert(key, value)
        return True
    
    def lookup(self, key: str) -> List[Any]:
        """查找键对应的值
        
        Args:
            key: 要查找的键
            
        Returns:
            匹配的值列表
        """
        if self.use_hash_index:
            return self.hash_index.lookup(key)
        else:
            # 回滚到线性扫描
            if key in self.fallback_data:
                return [self.fallback_data[key]]
            return []
    
    def remove(self, key: str) -> bool:
        """删除键值对
        
        Args:
            key: 要删除的键
            
        Returns:
            是否删除成功
        """
        # 从回滚数据中删除
        if key in self.fallback_data:
            del self.fallback_data[key]
        
        # 如果使用哈希索引，也从哈希索引中删除
        if self.use_hash_index:
            return self.hash_index.remove(key)
        return True
    
    def contains(self, key: str) -> bool:
        """检查键是否存在
        
        Args:
            key: 要检查的键
            
        Returns:
            是否存在
        """
        if self.use_hash_index:
            return self.hash_index.contains(key)
        else:
            return key in self.fallback_data
    
    def size(self) -> int:
        """获取索引中的键值对数量
        
        Returns:
            键值对数量
        """
        if self.use_hash_index:
            return self.hash_index.size()
        else:
            return len(self.fallback_data)
    
    def clear(self) -> None:
        """清空索引"""
        self.hash_index.clear()
        self.fallback_data.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'use_hash_index': self.use_hash_index,
            'fallback_size': len(self.fallback_data),
        }
        
        if self.use_hash_index:
            stats.update(self.hash_index.get_stats())
        
        return stats
    
    def enable_hash_index(self) -> None:
        """启用哈希索引"""
        if not self.use_hash_index:
            self.use_hash_index = True
            # 重建哈希索引
            self.hash_index.clear()
            for key, value in self.fallback_data.items():
                self.hash_index.insert(key, value)
    
    def disable_hash_index(self) -> None:
        """禁用哈希索引（回滚到线性扫描）"""
        self.use_hash_index = False
    
    def rebuild(self, new_table_size: Optional[int] = None) -> None:
        """重建哈希索引
        
        Args:
            new_table_size: 新的表大小
        """
        if self.use_hash_index:
            self.hash_index.rebuild(new_table_size)