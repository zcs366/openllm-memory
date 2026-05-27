"""Capsule哈希索引性能测试

测试不同规模数据下的性能表现。
"""

import tempfile
import time
import random
import string
import os
from openllm_memory import Capsule


def generate_random_string(length=10):
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def test_performance_at_scale(num_items, num_queries=100):
    """测试特定规模下的性能
    
    Args:
        num_items: 数据项数量
        num_queries: 查询次数
        
    Returns:
        性能统计字典
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试使用哈希索引
        capsule_with_hash = Capsule.open(os.path.join(tmpdir, "with_hash"), use_hash_index=True)
        
        # 生成测试数据
        keys = [f"key_{generate_random_string(20)}" for _ in range(num_items)]
        values = [f"value_{generate_random_string(20)}" for _ in range(num_items)]
        
        # 写入性能测试
        start_time = time.time()
        for key, value in zip(keys, values):
            capsule_with_hash.write(key, value)
        write_time_with_hash = time.time() - start_time
        
        # 检索性能测试（精确查找）
        query_keys = random.sample(keys, min(num_queries, len(keys)))
        start_time = time.time()
        for key in query_keys:
            capsule_with_hash.prefetch(key)
        exact_lookup_time_with_hash = time.time() - start_time
        
        # 检索性能测试（模糊查找）
        start_time = time.time()
        for i in range(num_queries):
            capsule_with_hash.prefetch(f"key_{i}")
        fuzzy_lookup_time_with_hash = time.time() - start_time
        
        # 测试不使用哈希索引
        capsule_without_hash = Capsule.open(os.path.join(tmpdir, "without_hash"), use_hash_index=False)
        
        # 写入性能测试
        start_time = time.time()
        for key, value in zip(keys, values):
            capsule_without_hash.write(key, value)
        write_time_without_hash = time.time() - start_time
        
        # 检索性能测试（精确查找）
        start_time = time.time()
        for key in query_keys:
            capsule_without_hash.prefetch(key)
        exact_lookup_time_without_hash = time.time() - start_time
        
        # 检索性能测试（模糊查找）
        start_time = time.time()
        for i in range(num_queries):
            capsule_without_hash.prefetch(f"key_{i}")
        fuzzy_lookup_time_without_hash = time.time() - start_time
        
        # 获取哈希索引统计
        stats = capsule_with_hash.get_hash_index_stats()
        
        return {
            'num_items': num_items,
            'num_queries': num_queries,
            'write_time_with_hash': write_time_with_hash,
            'write_time_without_hash': write_time_without_hash,
            'exact_lookup_time_with_hash': exact_lookup_time_with_hash,
            'exact_lookup_time_without_hash': exact_lookup_time_without_hash,
            'fuzzy_lookup_time_with_hash': fuzzy_lookup_time_with_hash,
            'fuzzy_lookup_time_without_hash': fuzzy_lookup_time_without_hash,
            'hash_index_stats': stats,
        }


def run_performance_tests():
    """运行性能测试"""
    print("开始运行Capsule哈希索引性能测试...\n")
    
    # 测试不同规模
    scales = [100, 500, 1000, 5000]
    
    results = []
    for scale in scales:
        print(f"测试规模: {scale} 项数据")
        result = test_performance_at_scale(scale, num_queries=50)
        results.append(result)
        
        # 输出结果
        print(f"  写入性能:")
        print(f"    使用哈希索引: {result['write_time_with_hash']:.3f}秒")
        print(f"    不使用哈希索引: {result['write_time_without_hash']:.3f}秒")
        print(f"    写入性能比: {result['write_time_with_hash']/result['write_time_without_hash']:.2f}x")
        
        print(f"  精确检索性能:")
        print(f"    使用哈希索引: {result['exact_lookup_time_with_hash']:.3f}秒")
        print(f"    不使用哈希索引: {result['exact_lookup_time_without_hash']:.3f}秒")
        print(f"    检索性能提升: {result['exact_lookup_time_without_hash']/result['exact_lookup_time_with_hash']:.2f}x")
        
        print(f"  模糊检索性能:")
        print(f"    使用哈希索引: {result['fuzzy_lookup_time_with_hash']:.3f}秒")
        print(f"    不使用哈希索引: {result['fuzzy_lookup_time_without_hash']:.3f}秒")
        print(f"    检索性能提升: {result['fuzzy_lookup_time_without_hash']/result['fuzzy_lookup_time_with_hash']:.2f}x")
        
        print(f"  哈希索引统计:")
        stats = result['hash_index_stats']
        print(f"    碰撞率: {stats['collision_rate']:.2%}")
        print(f"    负载因子: {stats['load_factor']:.2%}")
        print()
    
    # 生成报告
    print("\n" + "="*60)
    print("性能测试报告")
    print("="*60)
    
    print("\n写入性能:")
    print(f"{'规模':<10} {'使用哈希':<15} {'不使用哈希':<15} {'性能比':<10}")
    print("-" * 50)
    for result in results:
        ratio = result['write_time_with_hash'] / result['write_time_without_hash']
        print(f"{result['num_items']:<10} {result['write_time_with_hash']:.3f}秒{'':<10} {result['write_time_without_hash']:.3f}秒{'':<10} {ratio:.2f}x")
    
    print("\n精确检索性能:")
    print(f"{'规模':<10} {'使用哈希':<15} {'不使用哈希':<15} {'提升':<10}")
    print("-" * 50)
    for result in results:
        ratio = result['exact_lookup_time_without_hash'] / result['exact_lookup_time_with_hash']
        print(f"{result['num_items']:<10} {result['exact_lookup_time_with_hash']:.3f}秒{'':<10} {result['exact_lookup_time_without_hash']:.3f}秒{'':<10} {ratio:.2f}x")
    
    print("\n模糊检索性能:")
    print(f"{'规模':<10} {'使用哈希':<15} {'不使用哈希':<15} {'提升':<10}")
    print("-" * 50)
    for result in results:
        ratio = result['fuzzy_lookup_time_without_hash'] / result['fuzzy_lookup_time_with_hash']
        print(f"{result['num_items']:<10} {result['fuzzy_lookup_time_with_hash']:.3f}秒{'':<10} {result['fuzzy_lookup_time_without_hash']:.3f}秒{'':<10} {ratio:.2f}x")
    
    print("\n哈希索引统计:")
    print(f"{'规模':<10} {'碰撞率':<15} {'负载因子':<15}")
    print("-" * 40)
    for result in results:
        stats = result['hash_index_stats']
        print(f"{result['num_items']:<10} {stats['collision_rate']:.2%}{'':<10} {stats['load_factor']:.2%}")


if __name__ == "__main__":
    run_performance_tests()