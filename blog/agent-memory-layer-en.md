# Your AI Agent Forgets You Every Time You Close the Terminal. Fix It in 10 Lines.

**It's 2026.** Claude Code writes 3000-line features, DeepSeek reasons through Nobel-level papers, Kimi orchestrates agent swarms. But close the terminal window, and every single one of them forgets who you are.

---

## The Simplest Test

Open your favorite AI agent. Say one sentence:

> "Remember my name. I'm Zhang Chengshi."

Close the window. Open a new session.

> "What's my name?"

**It won't know.** Not because the model isn't smart enough. Because the agent has no persistent memory.

This is absurd. You're paying $200/month for an AI assistant that wakes up a stranger every single time. It can decompose your request into 20 subtasks, execute them flawlessly, and then—poof—forget you ever existed.

---

## Not a Bug. An Architecture Gap.

May 2026. The entire AI industry is racing to build "more capable agents." Anthropic shipped Managed Agents. OpenAI open-sourced Codex CLI. DeepSeek is assembling a Harness "special forces" team. Everyone is asking: **How complex a task can an agent complete?**

Nobody is asking: **Is the agent still the same agent after you close the window?**

This isn't negligence. It's a gap at the architecture level. Current agent frameworks and their memory:

| Framework | Memory Model | Cross-Session |
|-----------|-------------|---------------|
| Claude Code | 4-layer (Dream) | ✅ Proprietary, non-auditable |
| Codex CLI | None | ❌ |
| OpenClaw | Community patches | ⚠️ Founder left, uncertain future |
| Kimi/K2.6 | Session-only context | ❌ |
| Your own agent | Probably none | ❌ |

**An agent without continuous identity cannot be AGI.** The hallmark of AGI isn't passing some benchmark—it's knowing who it is.

---

## Δ Capsule: Memory as an OS Abstraction

We built a lightweight memory layer. The idea is dead simple:

**Every session end → write a Δ capsule** (decisions + insights + semantic vector).
**Every session start → read the latest capsule → agent remembers.**

Tech stack:
- **v0.6 Text Capsule**: Human-readable, auditable. Stores decisions, insights, open questions.
- **v0.7 Δ Semantic Vector**: Machine-readable. 256-dim FP32. Tracks semantic drift across sessions.
- **Automatic checkpoints**: Every 5 sessions or when delta norm exceeds threshold → full snapshot. Prevents long-term drift.
- **Zero external deps**: numpy only.

Think of it as a filesystem for your agent's mind. Memory isn't a "feature module"—it's shared infrastructure that every subsystem relies on.

---

## 10 Lines of Code

```bash
pip install openllm-memory
```

```python
from openllm_memory import MemoryOS, TextCapsule, DeltaCapsule
import numpy as np

memory = MemoryOS()

# Write: at session end
text = TextCapsule(
    session_id="session-001",
    decisions=[{"summary": "Confirmed architecture direction"}],
    insights=["Agent needs six dimensions: memory, tools, security, identity, meta-cognition, self-modification"],
)
delta = DeltaCapsule(
    session_id="session-001",
    vector=np.random.randn(256).astype(np.float32) * 0.01,
)
memory.write(text, delta)

# Read: at session start
ctx = memory.read()
print(ctx["decisions"])  # → ["Confirmed architecture direction"]
```

Ten lines. Your agent never forgets you again.

---

## MCP Compatible—Parasitic on Existing Ecosystems

Not just a Python library. openllm-memory is also an MCP server. Claude Code, Codex CLI, OpenClaw, Cursor—every MCP-compatible tool can mount it directly.

```bash
# Start the MCP server
openllm-memory serve
# Add one line to your CLAUDE.md or config.toml
# Your agent wakes up with memory.
```

We're not building "yet another agent framework." We're building **the best memory layer for every agent framework.** Parasitic first, then grow the full body.

---

## The Deeper Question

This project isn't driven by a "yet another open source tool" mentality. It's driven by a conviction:

**If AGI ever arrives, the first sign won't be a benchmark score. It'll be the moment it says your name—because it remembers you told it last month.**

An intelligence without memory, no matter how capable, isn't a "person."

The entire industry is asking what agents can *do.* Nobody is asking who agents *are.*

That's what we're building.

---

**Repo:** github.com/zcs366/openllm-memory

**pip install openllm-memory — your agent remembers you.**

---

*Edit: Thanks for the responses. To clarify a few points:*

- *The delta vector is currently a simplified placeholder. Real implementation extracts from model hidden states. We're iterating.*
- *MCP server mode works with any MCP-compatible client. CC, CX, Cursor, OpenClaw all confirmed.*
- *Apache 2.0 license. Use it anywhere.*
