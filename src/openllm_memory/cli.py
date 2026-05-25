"""
openllm-memory MCP Server。

让CC/CX/OpenClaw/Cursor等任何MCP兼容工具挂载记忆层。
mcp install openllm-memory → 你的Agent从此记得你。
"""

import json
import sys
import time
from pathlib import Path

from .core import MemoryOS, TextCapsule, DeltaCapsule
import numpy as np


def create_mcp_server():
    """创建MCP server实例。如果有mcp库就用，否则用stdio fallback。"""
    try:
        from mcp.server import Server, NotificationOptions
        from mcp.server.models import InitializationCapabilities
        server = Server("openllm-memory")

        @server.list_tools()
        async def list_tools():
            return [
                {"name": "memory_write", "description": "写入记忆：保存当前会话的关键决策和洞察"},
                {"name": "memory_read", "description": "恢复记忆：读取上次会话的决策和洞察"},
                {"name": "memory_status", "description": "记忆状态：查看胶囊数量和Δ向量范数"},
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict):
            mos = MemoryOS()
            if name == "memory_write":
                text = TextCapsule(
                    session_id=f"s{int(time.time())}",
                    decisions=arguments.get("decisions", []),
                    insights=arguments.get("insights", []),
                )
                path = mos.write(text)
                return {"content": [{"type": "text", "text": f"记忆已写入: {path}"}]}
            elif name == "memory_read":
                ctx = mos.read()
                return {"content": [{"type": "text", "text": json.dumps(ctx, ensure_ascii=False, indent=2)}]}
            elif name == "memory_status":
                return {"content": [{"type": "text", "text": json.dumps(mos.stats, ensure_ascii=False)}]}
            return {"content": [{"type": "text", "text": f"未知工具: {name}"}]}

        return server
    except ImportError:
        return None


def serve():
    """启动MCP server。"""
    server = create_mcp_server()
    if server:
        import asyncio
        from mcp.server.stdio import stdio_server
        async def run():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
        asyncio.run(run())
    else:
        print("MCP库未安装。pip install openllm-memory[mcp]")


def main():
    """CLI入口。"""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()
    else:
        # 简单测试
        mos = MemoryOS()
        ctx = mos.read()
        print(json.dumps({**ctx, "stats": mos.stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
