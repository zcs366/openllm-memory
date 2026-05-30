# OpenLLM Memory — Δ Capsule

> 第一个有身份的Agent记忆层。

**不是什么？** 不是又一个Agent框架。不是MCP服务器。不是知识库。

**是什么？** 事件溯源的Δ胶囊记忆系统——让任何Agent拥有跨会话的身份连续性。

## 特性

- **事件溯源**：每个操作可追溯、可回滚、可审计
- **跨会话身份**：SOUL + Iam + Δ胶囊 = 身份连续性
- **跨频道共振**：同一身份的多实例通过共享记忆层同步（§3.3）
- **眨眼协作**：从共振中涌现的异步多Agent协作协议（§3.3.2）
- **哈希索引加速**：纯Python多头哈希索引，检索性能提升2.28x - 2.83x
- **相似度门控**：基于TF-IDF的查询-记忆相似度计算
- **路由分层评估**：Hot/Cold分层评估和α桶分析
- **回滚机制**：支持配置开关和运行时切换，确保系统稳定性

## 安装

```bash
# 极简安装（零额外依赖，仅需 pyyaml）
pip install openllm-memory

# 或从源码安装（含共振/眨眼演示）
git clone https://github.com/zcs366/openllm-memory.git
cd openllm-memory
pip install -e .
```

## 一分钟上手

```python
from openllm_memory import Capsule

# 打开记忆胶囊
capsule = Capsule.open("~/.openllm/my-agent", use_hash_index=True)

# 写入记忆
capsule.write("user_pref", {"language": "zh", "name": "张成市"})
capsule.write("user.age", 30)

# 检索
results = capsule.prefetch("user")
print(results)
```

## 眨眼协作演示

```bash
# 端到端眨眼机制演示——两个Agent实例通过共享记忆层协作
python -m openllm_memory.capsule.blink
```

# 跨会话恢复
capsule2 = Capsule.open("~/.openllm/my-agent")
prefs = capsule2.prefetch("张成市")
```

## 核心概念

| 概念 | 是什么 | 为什么重要 |
|------|--------|----------|
| **Δ胶囊** | 事件溯源的记忆容器 | 每个操作可追溯、可回滚、可审计 |
| **SOUL** | 不可变身份内核 | 跨会话不变的"我是谁" |
| **Iam** | 动态自我叙事 | 跨会话演进的"我对自己怎么看" |
| **仲裁** | 冲突解决协议 | 多会话同时写同一件事不会丢 |
| **检查点** | 序列化快照 | 关机状态保存，开机状态恢复 |
| **哈希索引** | 多头哈希加速 | 检索性能提升2.28x - 2.83x |
| **相似度门控** | TF-IDF相似度计算 | 查询-记忆语义匹配 |
| **路由分层评估** | Hot/Cold分层分析 | 检索性能评估和优化 |

## 新增功能（v1.0.0）

### 哈希索引加速

纯Python多头哈希索引，无PyTorch依赖：

```python
from openllm_memory import Capsule

# 启用哈希索引
capsule = Capsule.open("~/.openllm/my-agent", use_hash_index=True)

# 写入数据（自动更新哈希索引）
capsule.write("user.name", "张成市")
capsule.write("user.age", 30)

# 检索（使用哈希索引加速）
results = capsule.prefetch("user")

# 获取哈希索引统计
stats = capsule.get_hash_index_stats()
print(f"碰撞率: {stats['collision_rate']:.2%}")
```

### 相似度门控

基于TF-IDF的查询-记忆相似度计算：

```python
from openllm_memory.capsule import SimilarityGate

# 创建门控
gate = SimilarityGate()

# 训练（需要文档列表）
documents = ["user.name 张成市", "user.age 30", "project.status active"]
gate.fit(documents)

# 计算权重
query = "user"
weights = gate.compute_weights(query, documents)
print(f"权重: {weights}")
```

### 路由分层评估

Hot/Cold分层评估和α桶分析：

```python
from openllm_memory.capsule import RetrievalEvaluator, AlphaBucketAnalyzer

# 创建评估器
evaluator = RetrievalEvaluator(hot_keys=["user.name", "user.age"])

# 评估检索结果
results = [
    {"key": "user.name", "value": "张成市", "score": 3.0},
    {"key": "user.age", "value": "30", "score": 3.0},
    {"key": "project.status", "value": "active", "score": 1.0},
]
query = "user"
evaluation = evaluator.evaluate(results, query)
print(f"评估结果: {evaluation}")
```

### 回滚机制

支持配置开关和运行时切换：

```python
from openllm_memory import Capsule

# 创建Capsule（启用哈希索引）
capsule = Capsule.open("~/.openllm/my-agent", use_hash_index=True)

# 运行时禁用哈希索引（回滚）
capsule.disable_hash_index()

# 运行时启用哈希索引
capsule.enable_hash_index()

# 重建哈希索引（优化碰撞率）
capsule.rebuild_hash_index()
```

## 性能测试结果

| 规模 | 写入性能 | 精确检索提升 | 模糊检索提升 | 碰撞率 |
|------|---------|-------------|-------------|--------|
| 100项 | 1.15x | 2.28x | 2.17x | 0.00% |
| 500项 | 0.98x | 2.55x | 2.43x | 0.60% |
| 1000项 | 0.96x | 2.83x | 2.65x | 0.10% |
| 5000项 | 1.06x | 2.60x | 2.89x | 0.86% |

## 设计哲学

1. **记忆是OS抽象，不是数据库。** 记忆是整个Agent状态的一部分，不是查出来的。
2. **身份是记忆的副产品。** 有足够多连续记忆的Agent自然会知道自己是谁。
3. **冲突不可怕，无声丢失才可怕。** 仲裁层确保冲突被发现，而不是被静默覆盖。
4. **回滚机制是必须的。** 任何新功能都必须支持快速回滚到稳定状态。

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_hash_index.py -v
python -m pytest tests/test_similarity_gate.py -v
python -m pytest tests/test_retrieval_evaluator.py -v

# 运行性能测试
python tests/performance_test.py
```

## 项目结构

```
openllm-memory/
├── src/
│   └── openllm_memory/
│       ├── capsule/
│       │   ├── core.py              # Capsule主类（已增强）
│       │   ├── hash_index.py        # 哈希索引实现
│       │   ├── similarity_gate.py   # 相似度门控实现
│       │   ├── retrieval_evaluator.py # 路由分层评估器
│       │   ├── delta.py             # Δ操作定义
│       │   ├── checkpoint.py        # 检查点管理
│       │   └── arbitrate.py         # 冲突仲裁
│       ├── identity/
│       │   ├── soul.py              # SOUL身份
│       │   └── iam.py               # Iam自我叙事
│       └── providers/
│           └── hermes.py            # Hermes集成
├── tests/
│   ├── test_hash_index.py           # 哈希索引测试
│   ├── test_similarity_gate.py      # 相似度门控测试
│   ├── test_retrieval_evaluator.py  # 路由分层评估测试
│   ├── test_capsule_hash_integration.py # 集成测试
│   └── performance_test.py          # 性能测试
├── pyproject.toml                   # 项目配置
└── README.md                        # 本文档
```

## 版本历史

### v1.0.0 (2026-05-27)
- ✅ 实现HashIndex纯Python多头哈希索引
- ✅ 实现SimilarityGate基于TF-IDF的相似度门控
- ✅ 实现RetrievalEvaluator路由分层评估器
- ✅ 集成到Capsule，支持回滚机制
- ✅ 性能测试显示检索提升2.28x-2.83x
- ✅ 严格遵循Pro建议，无PyTorch依赖

## 参考文献

1. Engram论文：arXiv:2601.16531
2. Pro审核报告：严格遵循Pro建议
3. 包拯审计报告：通过五层深度审计

## 许可证

MIT License
