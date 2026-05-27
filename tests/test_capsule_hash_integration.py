"""Capsule哈希索引集成测试

测试Capsule与HashIndex的集成功能，包括：
1. 基本读写操作
2. 哈希索引加速
3. 回滚机制
4. 性能对比
"""

import tempfile
import time
import os
from openllm_memory import Capsule


def test_basic_integration():
    """测试基本集成功能"""
    print("=== 测试基本集成 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建Capsule（启用哈希索引）
        capsule = Capsule.open(tmpdir, use_hash_index=True)
        
        # 写入数据
        capsule.write("user.name", "张成市")
        capsule.write("user.age", 30)
        capsule.write("project.status", "active")
        
        # 测试prefetch
        results = capsule.prefetch("user")
        print(f"prefetch 'user' 结果:\n{results}")
        
        # 验证结果包含预期内容
        assert "user.name" in results
        assert "user.age" in results
        
        # 测试哈希索引统计
        stats = capsule.get_hash_index_stats()
        print(f"哈希索引统计: {stats}")
        
        assert stats['use_hash_index'] == True
        assert stats['fallback_size'] == 3
        
        print("✓ 基本集成测试通过")


def test_rollback_mechanism():
    """测试回滚机制"""
    print("\n=== 测试回滚机制 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建Capsule（启用哈希索引）
        capsule = Capsule.open(tmpdir, use_hash_index=True)
        
        # 写入数据
        capsule.write("user.name", "张成市")
        capsule.write("user.age", 30)
        
        # 验证哈希索引工作
        results_with_hash = capsule.prefetch("user")
        print(f"使用哈希索引的结果:\n{results_with_hash}")
        
        # 禁用哈希索引（回滚）
        capsule.disable_hash_index()
        
        # 验证回滚后仍然工作
        results_without_hash = capsule.prefetch("user")
        print(f"回滚后结果:\n{results_without_hash}")
        
        # 结果应该相同
        assert "user.name" in results_without_hash
        assert "user.age" in results_without_hash
        
        # 重新启用哈希索引
        capsule.enable_hash_index()
        
        # 验证重新启用后工作
        results_re_enabled = capsule.prefetch("user")
        print(f"重新启用后结果:\n{results_re_enabled}")
        
        assert "user.name" in results_re_enabled
        assert "user.age" in results_re_enabled
        
        print("✓ 回滚机制测试通过")


def test_performance_comparison():
    """测试性能对比"""
    print("\n=== 测试性能对比 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试数据量
        num_items = 1000
        
        # 测试使用哈希索引的性能
        capsule_with_hash = Capsule.open(os.path.join(tmpdir, "with_hash"), use_hash_index=True)
        
        # 写入数据
        start_time = time.time()
        for i in range(num_items):
            capsule_with_hash.write(f"key_{i}", f"value_{i}")
        write_time_with_hash = time.time() - start_time
        
        # 测试检索性能
        start_time = time.time()
        for i in range(100):  # 测试100次检索
            capsule_with_hash.prefetch(f"key_{i}")
        lookup_time_with_hash = time.time() - start_time
        
        # 测试不使用哈希索引的性能
        capsule_without_hash = Capsule.open(os.path.join(tmpdir, "without_hash"), use_hash_index=False)
        
        # 写入数据
        start_time = time.time()
        for i in range(num_items):
            capsule_without_hash.write(f"key_{i}", f"value_{i}")
        write_time_without_hash = time.time() - start_time
        
        # 测试检索性能
        start_time = time.time()
        for i in range(100):  # 测试100次检索
            capsule_without_hash.prefetch(f"key_{i}")
        lookup_time_without_hash = time.time() - start_time
        
        # 输出结果
        print(f"写入 {num_items} 个item:")
        print(f"  使用哈希索引: {write_time_with_hash:.3f}秒")
        print(f"  不使用哈希索引: {write_time_without_hash:.3f}秒")
        print(f"  写入性能差异: {write_time_with_hash/write_time_without_hash:.2f}x")
        
        print(f"\n检索 100 个item:")
        print(f"  使用哈希索引: {lookup_time_with_hash:.3f}秒")
        print(f"  不使用哈希索引: {lookup_time_without_hash:.3f}秒")
        print(f"  检索性能提升: {lookup_time_without_hash/lookup_time_with_hash:.2f}x")
        
        # 获取哈希索引统计
        stats = capsule_with_hash.get_hash_index_stats()
        print(f"\n哈希索引统计:")
        print(f"  碰撞率: {stats['collision_rate']:.2%}")
        print(f"  负载因子: {stats['load_factor']:.2%}")
        
        # 性能应该有所提升
        # 注意：由于测试数据量小，提升可能不明显
        assert lookup_time_with_hash <= lookup_time_without_hash * 2  # 允许2倍误差
        
        print("✓ 性能对比测试通过")


def test_data_consistency():
    """测试数据一致性"""
    print("\n=== 测试数据一致性 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建Capsule
        capsule = Capsule.open(tmpdir, use_hash_index=True)
        
        # 写入数据
        capsule.write("user.name", "张成市")
        capsule.write("user.age", 30)
        capsule.write("user.email", "zhang@example.com")
        
        # 获取状态
        state = capsule.state()
        print(f"Capsule状态: {state}")
        
        # 验证哈希索引中的数据
        stats = capsule.get_hash_index_stats()
        print(f"哈希索引统计: {stats}")
        
        # 验证数据一致性
        assert state.get("user", {}).get("name") == "张成市"
        assert state.get("user", {}).get("age") == 30
        assert state.get("user", {}).get("email") == "zhang@example.com"
        
        # 验证哈希索引大小
        assert stats['fallback_size'] == 3
        
        print("✓ 数据一致性测试通过")


def test_rebuild_index():
    """测试重建索引"""
    print("\n=== 测试重建索引 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建Capsule
        capsule = Capsule.open(tmpdir, use_hash_index=True)
        
        # 写入数据
        for i in range(100):
            capsule.write(f"key_{i}", f"value_{i}")
        
        # 获取初始统计
        initial_stats = capsule.get_hash_index_stats()
        print(f"初始统计: {initial_stats}")
        
        # 重建索引
        capsule.rebuild_hash_index()
        
        # 获取重建后统计
        rebuilt_stats = capsule.get_hash_index_stats()
        print(f"重建后统计: {rebuilt_stats}")
        
        # 验证数据仍然可访问
        for i in range(100):
            results = capsule.prefetch(f"key_{i}")
            assert f"key_{i}" in results
        
        # 验证表大小增加
        assert rebuilt_stats['table_size'] > initial_stats['table_size']
        
        print("✓ 重建索引测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行Capsule哈希索引集成测试...\n")
    
    try:
        test_basic_integration()
        test_rollback_mechanism()
        test_performance_comparison()
        test_data_consistency()
        test_rebuild_index()
        
        print("\n" + "="*50)
        print("✓ 所有集成测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()