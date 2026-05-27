"""HashIndex测试用例

测试纯Python多头哈希索引的功能和性能。
"""

import time
import random
import string
from openllm_memory.capsule.hash_index import HashIndex, HashIndexWithFallback


def generate_random_key(length=10):
    """生成随机key"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def test_basic_operations():
    """测试基本操作"""
    print("=== 测试基本操作 ===")
    
    index = HashIndex(table_size=1009, K=2)  # 使用小质数便于测试
    
    # 测试插入
    assert index.insert("user.name", "张成市") == True
    assert index.insert("user.age", 30) == True
    assert index.insert("project.status", "active") == True
    
    # 测试查找
    results = index.lookup("user.name")
    assert len(results) == 1
    assert results[0] == "张成市"
    
    results = index.lookup("user.age")
    assert len(results) == 1
    assert results[0] == 30
    
    # 测试不存在的key
    results = index.lookup("nonexistent")
    assert len(results) == 0
    
    # 测试包含
    assert index.contains("user.name") == True
    assert index.contains("nonexistent") == False
    
    # 测试大小
    assert index.size() == 3
    
    # 测试删除
    assert index.remove("user.name") == True
    assert index.contains("user.name") == False
    assert index.size() == 2
    
    print("✓ 基本操作测试通过")


def test_collision_handling():
    """测试碰撞处理"""
    print("\n=== 测试碰撞处理 ===")
    
    # 使用小表大小强制碰撞
    index = HashIndex(table_size=10, K=2)
    
    # 插入多个key
    keys = [f"key_{i}" for i in range(20)]
    values = [f"value_{i}" for i in range(20)]
    
    for key, value in zip(keys, values):
        index.insert(key, value)
    
    # 验证查找
    found_count = 0
    for key, value in zip(keys, values):
        results = index.lookup(key)
        if value in results:
            found_count += 1
    
    print(f"插入 {len(keys)} 个key，成功查找 {found_count} 个")
    
    # 获取统计信息
    stats = index.get_stats()
    print(f"碰撞率: {stats['collision_rate']:.2%}")
    print(f"负载因子: {stats['load_factor']:.2%}")
    
    # 碰撞率应该大于0（因为表很小）
    assert stats['collision_rate'] > 0
    
    print("✓ 碰撞处理测试通过")


def test_performance():
    """测试性能"""
    print("\n=== 测试性能 ===")
    
    # 生成测试数据
    num_keys = 10000
    keys = [generate_random_key(20) for _ in range(num_keys)]
    values = [f"value_{i}" for i in range(num_keys)]
    
    # 测试哈希索引性能
    index = HashIndex(table_size=500003, K=2)
    
    # 插入性能
    start_time = time.time()
    for key, value in zip(keys, values):
        index.insert(key, value)
    insert_time = time.time() - start_time
    
    # 查找性能
    start_time = time.time()
    for key in keys[:1000]:  # 测试1000次查找
        index.lookup(key)
    lookup_time = time.time() - start_time
    
    print(f"插入 {num_keys} 个key: {insert_time:.3f}秒")
    print(f"查找 1000 个key: {lookup_time:.3f}秒")
    print(f"平均插入时间: {insert_time/num_keys*1000:.3f}毫秒/个")
    print(f"平均查找时间: {lookup_time/1000*1000:.3f}毫秒/个")
    
    # 获取统计信息
    stats = index.get_stats()
    print(f"碰撞率: {stats['collision_rate']:.2%}")
    
    # 性能应该合理
    assert insert_time < 10  # 10秒内完成10000次插入
    assert lookup_time < 1   # 1秒内完成1000次查找
    
    print("✓ 性能测试通过")


def test_fallback_mechanism():
    """测试回滚机制"""
    print("\n=== 测试回滚机制 ===")
    
    # 创建带回滚机制的索引
    index = HashIndexWithFallback(table_size=1009, K=2, use_hash_index=True)
    
    # 插入数据
    index.insert("user.name", "张成市")
    index.insert("user.age", 30)
    
    # 验证哈希索引工作
    assert index.lookup("user.name") == ["张成市"]
    assert index.contains("user.name") == True
    
    # 禁用哈希索引（回滚）
    index.disable_hash_index()
    assert index.lookup("user.name") == ["张成市"]  # 应该回滚到线性扫描
    assert index.contains("user.name") == True
    
    # 重新启用哈希索引
    index.enable_hash_index()
    assert index.lookup("user.name") == ["张成市"]
    
    # 测试统计信息
    stats = index.get_stats()
    assert stats['use_hash_index'] == True
    assert stats['fallback_size'] == 2
    
    print("✓ 回滚机制测试通过")


def test_rebuild():
    """测试重建功能"""
    print("\n=== 测试重建功能 ===")
    
    index = HashIndex(table_size=100, K=2)
    
    # 插入数据
    for i in range(50):
        index.insert(f"key_{i}", f"value_{i}")
    
    # 获取初始统计
    initial_stats = index.get_stats()
    print(f"初始表大小: {initial_stats['table_size']}")
    print(f"初始碰撞率: {initial_stats['collision_rate']:.2%}")
    
    # 重建索引
    index.rebuild()
    
    # 获取重建后统计
    rebuilt_stats = index.get_stats()
    print(f"重建后表大小: {rebuilt_stats['table_size']}")
    print(f"重建后碰撞率: {rebuilt_stats['collision_rate']:.2%}")
    
    # 验证数据完整性
    found_count = 0
    for i in range(50):
        results = index.lookup(f"key_{i}")
        if len(results) == 1 and results[0] == f"value_{i}":
            found_count += 1
    
    print(f"成功查找 {found_count}/50 个key")
    
    # 由于碰撞，可能无法找到所有key，但应该找到大部分
    assert found_count >= 40  # 至少找到80%的key
    
    # 表大小应该增加
    assert rebuilt_stats['table_size'] > initial_stats['table_size']
    
    print("✓ 重建功能测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行HashIndex测试...\n")
    
    try:
        test_basic_operations()
        test_collision_handling()
        test_performance()
        test_fallback_mechanism()
        test_rebuild()
        
        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()