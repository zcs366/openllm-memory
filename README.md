# openllm-memory — Δ胶囊·ISA Project的记忆固化系统

**pip install openllm-memory，你的Agent从此不再从零学起。**

Δ胶囊是ISA Project这个人工认知架构的第三层——记忆固化系统（群体学习层）。
前两层：ISA Layer(神经纤维·感知运动) + Brain Layer(大脑皮层·个体认知)。

任何Agent框架（Hermes、OpenClaw、CC via MCP）只需挂载这一层，立即获得：
- 跨会话连续身份（Agent关掉窗口后不再失忆）
- 跨Agent认知查询（"谁梦到过什么？"）
- 群体记忆固化（个体洞察 → 共享知识）

## 在ISA Project中的位置

```
ISA Project = 人工认知架构
├── ISA Layer     = 神经纤维     → gateway.py + 波扩散    — 感知运动
├── Brain Layer   = 大脑皮层     → brain.py               — 个体认知
└── Δ胶囊 Layer   = 记忆固化系统  → openllm-memory         — 群体记忆 ← 本包
```

## 核心接口

```python
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule, DreamBridge

# 写入记忆
memory = MemoryOS()
capsule = TextCapsule(session_id="s-001", insights=["..."])
memory.write(capsule)

# 消费ISA Brain的梦境日志（三角闭环的关键桥梁）
bridge = DreamBridge()
result = bridge.consume()  # 增量读取dream事件→转胶囊→存入MemoryOS
bridge.query("ISA和jika的关系")  # "谁梦到过这个？"
bridge.hot_topics()            # "本周最热的认知关联"

# 恢复记忆
ctx = memory.read()
```

## MCP Server模式

```bash
# 启动MCP server
openllm-memory serve

# 其他MCP兼容工具直接挂载
# CC: 在CLAUDE.md中配置MCP server
# CX: 在~/.codex/config.toml中配置
# OpenClaw: 在skills中引用
```

## API

| 方法 | 说明 |
|------|------|
| `MemoryOS()` | 初始化记忆OS |
| `.write(text, delta)` | 写入双胶囊+自动检查点 |
| `.read()` | 恢复最新记忆 |
| `.checkpoint()` | 手动触发全量快照 |
| `.capsule_dir` | 胶囊存储路径 |
