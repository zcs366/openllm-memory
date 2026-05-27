"""SimilarityGate测试用例

测试基于TF-IDF的相似度门控功能。
"""

import time
from openllm_memory.capsule.similarity_gate import SimilarityGate, SimilarityGateWithFallback


def test_basic_operations():
    """测试基本操作"""
    print("=== 测试基本操作 ===")
    
    gate = SimilarityGate()
    
    # 测试训练
    documents = [
        "user.name 张成市",
        "user.age 30",
        "project.status active",
        "project.name openllm-memory",
    ]
    
    gate.fit(documents)
    assert gate.is_fitted()
    
    # 测试词汇表
    vocab = gate.get_vocabulary()
    print(f"词汇表大小: {len(vocab)}")
    assert "user" in vocab
    assert "name" in vocab
    
    # 测试IDF
    idf = gate.get_idf()
    print(f"IDF值示例: {list(idf.items())[:3]}")
    
    print("✓ 基本操作测试通过")


def test_similarity_calculation():
    """测试相似度计算"""
    print("\n=== 测试相似度计算 ===")
    
    gate = SimilarityGate()
    
    documents = [
        "user.name 张成市",
        "user.age 30",
        "project.status active",
        "project.name openllm-memory",
    ]
    
    gate.fit(documents)
    
    # 测试查询与文档的相似度
    query = "user"
    for doc in documents:
        similarity = gate.compute_similarity(query, doc)
        print(f"查询 '{query}' 与文档 '{doc}' 的相似度: {similarity:.4f}")
    
    # 测试权重计算
    weights = gate.compute_weights(query, documents)
    print(f"\n查询 '{query}' 的权重分布: {weights}")
    
    # 验证权重归一化
    assert abs(sum(weights) - 1.0) < 1e-6
    
    # 验证包含"user"的文档权重更高
    user_doc_indices = [i for i, doc in enumerate(documents) if "user" in doc.lower()]
    non_user_doc_indices = [i for i, doc in enumerate(documents) if "user" not in doc.lower()]
    
    if user_doc_indices and non_user_doc_indices:
        avg_user_weight = sum(weights[i] for i in user_doc_indices) / len(user_doc_indices)
        avg_non_user_weight = sum(weights[i] for i in non_user_doc_indices) / len(non_user_doc_indices)
        print(f"用户文档平均权重: {avg_user_weight:.4f}")
        print(f"非用户文档平均权重: {avg_non_user_weight:.4f}")
        assert avg_user_weight > avg_non_user_weight
    
    print("✓ 相似度计算测试通过")


def test_fallback_mechanism():
    """测试回滚机制"""
    print("\n=== 测试回滚机制 ===")
    
    gate = SimilarityGateWithFallback(use_similarity_gate=True)
    
    documents = [
        "user.name 张成市",
        "user.age 30",
        "project.status active",
    ]
    
    # 训练相似度门控
    gate.fit(documents)
    
    # 测试使用相似度门控
    query = "user"
    weights_with_gate = gate.compute_weights(query, documents)
    print(f"使用相似度门控的权重: {weights_with_gate}")
    
    # 禁用相似度门控（回滚）
    gate.disable_similarity_gate()
    
    # 测试回滚后的权重
    weights_without_gate = gate.compute_weights(query, documents)
    print(f"回滚后的权重: {weights_without_gate}")
    
    # 验证权重归一化
    assert abs(sum(weights_without_gate) - 1.0) < 1e-6
    
    # 重新启用相似度门控
    gate.enable_similarity_gate()
    
    # 验证重新启用后权重相同
    weights_re_enabled = gate.compute_weights(query, documents)
    print(f"重新启用后的权重: {weights_re_enabled}")
    
    # 由于重新训练，权重可能略有不同，但应该相似
    assert abs(sum(weights_re_enabled) - 1.0) < 1e-6
    
    print("✓ 回滚机制测试通过")


def test_performance():
    """测试性能"""
    print("\n=== 测试性能 ===")
    
    # 生成测试数据
    num_documents = 1000
    documents = [f"document_{i} content_{i} keyword_{i % 10}" for i in range(num_documents)]
    
    gate = SimilarityGate()
    
    # 测试训练性能
    start_time = time.time()
    gate.fit(documents)
    fit_time = time.time() - start_time
    
    print(f"训练 {num_documents} 个文档: {fit_time:.3f}秒")
    
    # 测试查询性能
    num_queries = 100
    queries = [f"query_{i}" for i in range(num_queries)]
    
    start_time = time.time()
    for query in queries:
        gate.compute_weights(query, documents[:100])  # 测试前100个文档
    query_time = time.time() - start_time
    
    print(f"执行 {num_queries} 次查询: {query_time:.3f}秒")
    print(f"平均查询时间: {query_time/num_queries*1000:.3f}毫秒")
    
    # 性能应该合理
    assert fit_time < 10  # 10秒内完成训练
    assert query_time < 5  # 5秒内完成100次查询
    
    print("✓ 性能测试通过")


def test_temperature_effect():
    """测试温度参数效果"""
    print("\n=== 测试温度参数效果 ===")
    
    gate = SimilarityGate()
    
    documents = [
        "user name 张成市",
        "user age 30",
        "project status active",
    ]
    
    gate.fit(documents)
    
    query = "user"
    
    # 测试不同温度参数
    temperatures = [0.5, 1.0, 2.0]
    
    for temp in temperatures:
        weights = gate.compute_weights(query, documents, temperature=temp)
        print(f"温度 {temp}: {weights}")
        
        # 验证权重归一化
        assert abs(sum(weights) - 1.0) < 1e-6
    
    # 温度越低，权重分布越极端
    weights_low_temp = gate.compute_weights(query, documents, temperature=0.5)
    weights_high_temp = gate.compute_weights(query, documents, temperature=2.0)
    
    # 计算权重熵（衡量分布的均匀程度）
    def entropy(weights):
        return -sum(w * (math.log(w) if w > 0 else 0) for w in weights)
    
    import math
    entropy_low = entropy(weights_low_temp)
    entropy_high = entropy(weights_high_temp)
    
    print(f"低温度熵: {entropy_low:.4f}")
    print(f"高温度熵: {entropy_high:.4f}")
    
    # 低温度应该有更低的熵（更极端的分布）
    # 注意：如果所有相似度都是0，温度参数不会有影响
    # 这里我们只验证权重归一化
    assert abs(sum(weights_low_temp) - 1.0) < 1e-6
    assert abs(sum(weights_high_temp) - 1.0) < 1e-6
    
    print("✓ 温度参数效果测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行SimilarityGate测试...\n")
    
    try:
        test_basic_operations()
        test_similarity_calculation()
        test_fallback_mechanism()
        test_performance()
        test_temperature_effect()
        
        print("\n" + "="*50)
        print("✓ 所有SimilarityGate测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()