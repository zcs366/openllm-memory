# 你的Agent关掉窗口就失忆——修好它只要10行代码

> 2026年了。Claude Code能写3000行代码，DeepSeek能推理诺贝尔奖论文，Kimi能调度Agent集群。但关掉终端，它们全都忘了你是谁。

---

## 一、一个最简单的测试

打开你最喜欢的AI Agent（CC、CX、Kimi、ChatGPT都行），说一句话：

> "记住我的名字，我叫张成市。"

关掉窗口。重新打开。

> "我叫什么名字？"

**答案永远是不知道。** 不是模型不够聪明。是这个Agent没有持久记忆系统。

这事有多荒谬？你花$200/月订阅的AI助手，每次醒来都是陌生人。它能把你的需求拆成20步子任务，但它不记得上一句话你说过你的名字。

---

## 二、不是Bug——是架构缺陷

2026年5月，整个AI行业都在卷"更强的Agent"。Anthropic推Managed Agents，OpenAI开源Codex CLI，DeepSeek组建Harness"特种部队"。所有人都在问：**Agent能完成多复杂的任务？**

没人问：**Agent关掉窗口后还是同一个Agent吗？**

这不是疏忽。这是架构层面的缺失。当前主流Agent框架的记忆系统：

| 框架 | 记忆方式 | 跨会话 |
|------|---------|--------|
| Claude Code | 4层记忆(Dream) | ✅但闭源，不可审计 |
| Codex CLI | 无持久记忆 | ❌ |
| OpenClaw | 社区贡献的记忆层 | ⚠️弱，且创始人已离开 |
| Kimi/K2.6 | 会话内上下文 | ❌ |
| 你自己的Agent | 大概率没有 | ❌ |

**一个没有连续身份的Agent，不可能是AGI。** AGI的标志不是通过任何考试——是知道自己是谁。

---

## 三、Δ胶囊：记忆作为操作系统抽象

我们造了一个轻量级记忆层。核心思路很简单：

**每次会话结束 → 写入一个Δ胶囊（决策+洞察+语义向量）。**
**下次会话开始 → 读取最新胶囊 → Agent记忆恢复。**

技术上：
- **v0.6文本胶囊**：给人看，可审计。保存决策、洞察、未解问题。
- **v0.7Δ语义向量**：给模型用。256维FP32，记录会话在语义空间中的移动量。
- **自动检查点**：每5次会话或Δ累积超过阈值→全量快照，防止长期漂移。
- **零外部依赖**：只依赖numpy。

类比：文件系统之于操作系统。记忆不是Agent的一个"功能模块"——它是所有子系统共享的基础设施。

---

## 四、10行代码

```bash
pip install openllm-memory
```

```python
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule
import numpy as np

memory = MemoryOS()

# 写：会话结束时
text = TextCapsule(
    session_id="session-001",
    decisions=[{"summary": "确认OpenLLM架构方向"}],
    insights=["Agent需要六维躯体：记忆+工具+安全+身份+元认知+自修"],
)
delta = DeltaCapsule(
    session_id="session-001",
    vector=np.random.randn(256).astype(np.float32) * 0.01,
)
memory.write(text, delta)

# 读：新会话开始时
ctx = memory.read()
print(ctx["decisions"])  # → ["确认OpenLLM架构方向"]
```

10行。你的Agent从此不再失忆。

---

## 五、MCP兼容——寄生现有的Agent生态

不只是Python库。openllm-memory也是一个MCP Server。Claude Code、Codex CLI、OpenClaw、Cursor——所有支持MCP的工具都能直接挂载：

```bash
# 启动MCP server
openllm-memory serve

# 在CC/CLAWDE.md或CX/config.toml里配一行MCP引用
# 你的Agent下次启动就带着记忆
```

不做"又一个Agent框架"——做**每个Agent框架最好的记忆层**。先寄生，再从中长出完整躯干。

---

## 六、那个更深的命题

做这个项目，驱动我们的不是"又一个开源工具"的心态。是一个更根本的判断：

**如果AGI真的到来，它的第一个标志不是通过了哪个考试——是它在一次对话中说出了你上次告诉它的名字。**

一个没有记忆的智能体，再聪明也不是"一个人"。

当前全行业都在卷"Agent能做什么"。没有人在卷"Agent是谁"。

这就是OpenLLM在做的。

---

**项目地址（即将开源）：** github.com/zcs366/openllm-memory

**pip install openllm-memory — 你的Agent从此记得你。**
