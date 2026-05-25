# openllm-memory — Δ胶囊记忆层

**pip install openllm-memory，你的Agent从此记得用户。**

任何Agent框架（Hermes、OpenClaw、CC via MCP、CX via MCP、自建）只需挂载这一层，立即获得跨会话连续身份。

## 三句话

- **记忆不是功能模块——是操作系统抽象。** 像文件系统之于程序。
- **你的Agent关掉窗口后不再失忆。** 下次醒来认得老搭档。
- **任何框架都能用。** 不是"又一个框架"——是"每个框架最好的记忆层"。

## 安装

```bash
pip install openllm-memory

# MCP server模式（让CC/CX/OpenClaw都能用）
pip install openllm-memory[mcp]
```

## 使用

```python
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule
import numpy as np

# 初始化
memory = MemoryOS()

# 写入记忆
text = TextCapsule(
    session_id="session-001",
    decisions=[{"summary": "确认架构方向"}],
    insights=["Agent需要六维躯体"],
)
delta = DeltaCapsule(
    session_id="session-001",
    vector=np.random.randn(256).astype(np.float32) * 0.01,
)
memory.write(text, delta)

# 恢复记忆
ctx = memory.read()
print(ctx["decisions"])  # ["确认架构方向"]
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
