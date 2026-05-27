"""RetrievalEvaluator测试用例

测试路由分层评估器和α桶分析功能。
"""

import time
from openllm_memory.capsule.retrieval_evaluator import RetrievalEvaluator, AlphaBucketAnalyzer, RetrievalEvaluatorWithFallback


def test_basic_evaluation():
    """测试基本评估功能"""
    print("=== 测试基本评估 ===")
    
    evaluator = RetrievalEvaluator(hot_keys=["user.name", "user.age"])
    
    # 模拟检索结果
    results = [
        {"key": "user.name", "value": "张成市", "score": 3.0},
        {"key": "user.age", "value": "30", "score": 3.0},
        {"key": "project.status", "value": "active", "score": 1.0},
        {"key": "project.name", "value": "openllm-memory", "score": 0.0},
    ]
    
    query = "user"
    
    # 执行评估
    evaluation = evaluator.evaluate(results, query)
    
    print(f"评估结果: {evaluation}")
    
    # 验证结果
    assert evaluation['hot_count'] == 2  # user.name 和 user.age
    assert evaluation['cold_count'] == 2  # project.status 和 project.name
    assert evaluation['total_count'] == 4
    
    # 验证热键平均分数更高
    assert evaluation['hot_mean'] > evaluation['cold_mean']
    
    print("✓ 基本评估测试通过")


def test_alpha_bucket_analysis():
    """测试α桶分析"""
    print("\n=== 测试α桶分析 ===")
    
    analyzer = AlphaBucketAnalyzer()
    
    # 模拟权重和分数
    weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    scores = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    
    # 执行分析
    analysis = analyzer.analyze(weights, scores)
    
    print(f"分析结果:")
    for bucket, stats in analysis.items():
        if bucket != 'overall':
            print(f"  {bucket}: 平均分={stats['mean_score']:.2f}, 数量={stats['count']}")
    
    print(f"  总体: 平均分={analysis['overall']['mean_score']:.2f}, 数量={analysis['overall']['count']}")
    
    # 验证结果
    assert analysis['0.0-0.2']['count'] == 2  # 0.0 和 0.1
    assert analysis['0.8-1.0']['count'] == 2  # 0.8 和 0.9
    
    # 验证高权重桶有更高的平均分
    assert analysis['0.8-1.0']['mean_score'] > analysis['0.0-0.2']['mean_score']
    
    print("✓ α桶分析测试通过")


def test_fallback_mechanism():
    """测试回滚机制"""
    print("\n=== 测试回滚机制 ===")
    
    evaluator = RetrievalEvaluatorWithFallback(
        hot_keys=["user.name", "user.age"],
        use_detailed_evaluation=True
    )
    
    # 模拟检索结果
    results = [
        {"key": "user.name", "value": "张成市", "score": 3.0},
        {"key": "user.age", "value": "30", "score": 3.0},
        {"key": "project.status", "value": "active", "score": 1.0},
    ]
    
    query = "user"
    
    # 测试详细评估
    detailed_evaluation = evaluator.evaluate(results, query)
    print(f"详细评估: {detailed_evaluation}")
    
    # 禁用详细评估（回滚）
    evaluator.disable_detailed_evaluation()
    
    # 测试简单评估
    simple_evaluation = evaluator.evaluate(results, query)
    print(f"简单评估: {simple_evaluation}")
    
    # 验证简单评估不区分热冷
    assert simple_evaluation['hot_count'] == 0
    assert simple_evaluation['cold_count'] == len(results)
    
    # 重新启用详细评估
    evaluator.enable_detailed_evaluation()
    
    # 验证重新启用后结果相同
    re_enabled_evaluation = evaluator.evaluate(results, query)
    print(f"重新启用后评估: {re_enabled_evaluation}")
    
    assert re_enabled_evaluation['hot_count'] == detailed_evaluation['hot_count']
    
    print("✓ 回滚机制测试通过")


def test_performance():
    """测试性能"""
    print("\n=== 测试性能 ===")
    
    # 生成测试数据
    num_results = 1000
    results = []
    for i in range(num_results):
        results.append({
            "key": f"key_{i}",
            "value": f"value_{i}",
            "score": float(i % 10),
        })
    
    # 设置热键
    hot_keys = [f"key_{i}" for i in range(0, num_results, 10)]  # 每10个key中一个热键
    
    evaluator = RetrievalEvaluator(hot_keys=hot_keys)
    query = "key"
    
    # 测试评估性能
    start_time = time.time()
    for _ in range(100):  # 测试100次评估
        evaluator.evaluate(results, query)
    evaluation_time = time.time() - start_time
    
    print(f"评估 {num_results} 个结果 100 次: {evaluation_time:.3f}秒")
    print(f"平均评估时间: {evaluation_time/100*1000:.3f}毫秒")
    
    # 测试α桶分析性能
    analyzer = AlphaBucketAnalyzer()
    weights = [float(i) / num_results for i in range(num_results)]
    scores = [float(i % 10) for i in range(num_results)]
    
    start_time = time.time()
    for _ in range(100):  # 测试100次分析
        analyzer.analyze(weights, scores)
    analysis_time = time.time() - start_time
    
    print(f"α桶分析 {num_results} 个数据点 100 次: {analysis_time:.3f}秒")
    print(f"平均分析时间: {analysis_time/100*1000:.3f}毫秒")
    
    # 性能应该合理
    assert evaluation_time < 5  # 5秒内完成100次评估
    assert analysis_time < 5   # 5秒内完成100次分析
    
    print("✓ 性能测试通过")


def test_hot_key_management():
    """测试热键管理"""
    print("\n=== 测试热键管理 ===")
    
    evaluator = RetrievalEvaluator()
    
    # 测试添加热键
    evaluator.add_hot_key("user.name")
    evaluator.add_hot_key("user.age")
    hot_keys = evaluator.get_hot_keys()
    assert "user.name" in hot_keys
    assert "user.age" in hot_keys
    assert len(hot_keys) == 2
    
    # 测试移除热键
    evaluator.remove_hot_key("user.name")
    hot_keys = evaluator.get_hot_keys()
    assert "user.name" not in hot_keys
    assert "user.age" in hot_keys
    
    # 测试设置热键列表
    evaluator.set_hot_keys(["project.name", "project.status"])
    hot_keys = evaluator.get_hot_keys()
    assert "project.name" in hot_keys
    assert "project.status" in hot_keys
    
    print("✓ 热键管理测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    evaluator = RetrievalEvaluator(hot_keys=["user.name"])
    
    # 测试空结果
    empty_results = []
    evaluation = evaluator.evaluate(empty_results, "user")
    print(f"空结果评估: {evaluation}")
    
    assert evaluation['total_count'] == 0
    assert evaluation['hot_count'] == 0
    assert evaluation['cold_count'] == 0
    
    # 测试无匹配查询
    results = [
        {"key": "user.name", "value": "张成市"},  # 移除score字段
        {"key": "project.status", "value": "active"},  # 移除score字段
    ]
    
    evaluation = evaluator.evaluate(results, "nonexistent")
    print(f"无匹配查询评估: {evaluation}")
    
    # 由于查询"nonexistent"不在任何key或value中，分数应该为0
    assert evaluation['hot_mean'] == 0.0
    assert evaluation['cold_mean'] == 0.0
    
    # 测试α桶分析边界情况
    analyzer = AlphaBucketAnalyzer()
    
    # 测试空数据
    empty_analysis = analyzer.analyze([], [])
    print(f"空数据分析: {empty_analysis}")
    
    assert empty_analysis['overall']['count'] == 0
    
    print("✓ 边界情况测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行RetrievalEvaluator测试...\n")
    
    try:
        test_basic_evaluation()
        test_alpha_bucket_analysis()
        test_fallback_mechanism()
        test_performance()
        test_hot_key_management()
        test_edge_cases()
        
        print("\n" + "="*50)
        print("✓ 所有RetrievalEvaluator测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()