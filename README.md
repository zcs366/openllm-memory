# OpenLLM Memory — Δ Capsule

> 第一个有身份的Agent记忆层。

**不是什么？** 不是又一个Agent框架。不是MCP服务器。不是知识库。

**是什么？** 事件溯源的Δ胶囊记忆系统——让任何Agent拥有跨会话的身份连续性。

## 安装

```bash
pip install openllm-memory
```

## 一分钟上手

```python
from openllm_memory import Capsule

# 打开一个记忆胶囊
capsule = Capsule.open("~/.openllm/my-agent")

# 写一条记忆（一个Δ操作）
capsule.write("user_pref", {"language": "zh", "name": "张成市"})

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

## 设计哲学

1. **记忆是OS抽象，不是数据库。** 记忆是整个Agent状态的一部分，不是查出来的。
2. **身份是记忆的副产品。** 有足够多连续记忆的Agent自然会知道自己是谁。
3. **冲突不可怕，无声丢失才可怕。** 仲裁层确保冲突被发现，而不是被静默覆盖。

## Hermes集成

```bash
# 在Hermes配置中启用
pip install openllm-memory[hermes]
```
然后在 `config.yaml` 中：
```yaml
memory:
  provider: openllm-memory
```

## MCP集成

任何支持MCP的工具（Claude Code、Codex CLI、Cursor等）都能挂载：

```bash
pip install openllm-memory[mcp]
mcp install openllm-memory
```
