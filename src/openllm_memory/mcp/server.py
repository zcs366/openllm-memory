"""
MCP Server for openllm-memory

Exposes the Δ Capsule as an MCP (Model Context Protocol) server.
Any MCP-compatible tool (Claude Code, Codex CLI, Cursor, etc.)
can mount this and get cross-session memory.

Usage:
    pip install openllm-memory[mcp]
    mcp install /path/to/openllm-memory/src/openllm_memory/mcp/server.py
    # or:
    python -m openllm_memory.mcp.server --store ~/.openllm/capsule

Tools exposed:
    - capsule_write: Write a memory Δ
    - capsule_read: Read a memory key
    - capsule_prefetch: Search memories by keyword
    - capsule_whoami: Get agent identity (SOUL + Iam)
    - capsule_reflect: Update self-narrative
    - capsule_state: Get full memory state
    - capsule_compact: Compact Δ log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Try MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        TextContent,
        Tool,
    )
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from openllm_memory import Capsule, Soul, Iam
from openllm_memory.capsule import Arbitrator


def create_server(store_dir: str = "~/.openllm/capsule") -> Any:
    """Create an MCP server with Δ Capsule tools.
    
    Args:
        store_dir: Path to the capsule storage directory
    
    Returns:
        MCP Server instance
    """
    if not HAS_MCP:
        raise ImportError(
            "MCP SDK not installed. Run: pip install openllm-memory[mcp]"
        )
    
    store_path = str(Path(store_dir).expanduser().resolve())
    capsule = Capsule.open(store_path)
    
    # Load or create SOUL
    soul_path = Path(store_path) / "soul.json"
    if soul_path.exists():
        soul = Soul.load(str(soul_path))
    else:
        soul = Soul.create({"name": "MCP Agent", "type": "agent"})
        soul.save(str(soul_path))
    
    # Create Iam
    iam = Iam(str(Path(store_path) / "iam"), capsule)
    
    app = Server("openllm-memory")
    
    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="capsule_write",
                description="Write a memory Δ. Every memory write is an immutable event.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key (dot-path)"},
                        "value": {"description": "Value to store"},
                        "op": {"type": "string", "enum": ["set", "delete", "merge", "append"]},
                    },
                    "required": ["key", "value"],
                },
            ),
            Tool(
                name="capsule_read",
                description="Read a specific memory key",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key"},
                    },
                    "required": ["key"],
                },
            ),
            Tool(
                name="capsule_prefetch",
                description="Search memories by keyword. Returns relevant context.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="capsule_whoami",
                description="Get agent identity — SOUL (immutable) + Iam (self-narrative)",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="capsule_reflect",
                description="Update self-narrative (Iam)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "value": {"type": "object", "description": "Updated self-description"},
                        "reason": {"type": "string", "description": "Why this change"},
                    },
                    "required": ["value"],
                },
            ),
            Tool(
                name="capsule_state",
                description="Get the entire current memory state",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="capsule_compact",
                description="Compress Δ log into checkpoint",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]
    
    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        global capsule, soul, iam
        
        try:
            if name == "capsule_write":
                key = arguments["key"]
                value = arguments.get("value")
                op = arguments.get("op", "set")
                delta = capsule.write(key, value, op=op, session_id="mcp")
                return [TextContent(
                    type="text",
                    text=json.dumps({"success": True, "delta_id": delta.delta_id})
                )]
            
            elif name == "capsule_read":
                key = arguments["key"]
                value = capsule.get(key)
                return [TextContent(
                    type="text",
                    text=json.dumps({"key": key, "value": value}, ensure_ascii=False)
                )]
            
            elif name == "capsule_prefetch":
                query = arguments["query"]
                result = capsule.prefetch(query)
                return [TextContent(
                    type="text",
                    text=result or json.dumps({"note": "No relevant memories found"})
                )]
            
            elif name == "capsule_whoami":
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "soul": soul.to_dict(),
                        "iam": iam.all(),
                        "delta_count": capsule.delta_count(),
                    }, ensure_ascii=False)
                )]
            
            elif name == "capsule_reflect":
                value = arguments.get("value", {})
                reason = arguments.get("reason", "agent self-reflection")
                iam.update(value, reason=reason)
                return [TextContent(
                    type="text",
                    text=json.dumps({"success": True, "iam": iam.all()}, ensure_ascii=False)
                )]
            
            elif name == "capsule_state":
                state = capsule.state()
                return [TextContent(
                    type="text",
                    text=json.dumps(state, indent=2, ensure_ascii=False)
                )]
            
            elif name == "capsule_compact":
                count = capsule.compact()
                return [TextContent(
                    type="text",
                    text=json.dumps({"compacted": count, "note": "Δ log compressed to checkpoint"})
                )]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    return app


def main():
    """Run the MCP server from CLI."""
    store_dir = "~/.openllm/capsule"
    if "--store" in sys.argv:
        idx = sys.argv.index("--store")
        if idx + 1 < len(sys.argv):
            store_dir = sys.argv[idx + 1]
    
    if not HAS_MCP:
        print("MCP SDK not installed. Run: pip install openllm-memory[mcp]")
        sys.exit(1)
    
    import asyncio
    app = create_server(store_dir)
    asyncio.run(stdio_server(app))


if __name__ == "__main__":
    main()
