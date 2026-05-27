"""
openllm-memory Hermes MemoryProvider plugin

把Δ胶囊挂到Hermes上，成为唯一的记忆后端。
Makes Hermes have cross-session identity continuity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from openllm_memory import Capsule, Soul, Iam

logger = logging.getLogger(__name__)


CAPSULE_TOOL_SCHEMA = {
    "name": "capsule",
    "description": (
        "Δ Capsule — event-sourced, identity-driven memory. "
        "Write, read, and manage persistent memories that survive across sessions. "
        "Every write is a Δ (delta) that can be traced, rolled back, or reconciled.\\n\\n"
        "ACTIONS:\\n"
        "• write — Store a memory (key, value, op=set/delete/merge/append)\\n"
        "• read — Get a memory by key\\n"
        "• state — Get the entire current memory state\\n"
        "• prefetch — Search memory by keyword\\n"
        "• whoami — Get the agent's identity (SOUL + Iam)\\n"
        "• reflect — Update self-narrative (Iam)\\n"
        "• compact — Compress Δ log into checkpoint\\n\\n"
        "Unlike fact_store (structured facts), the capsule stores the agent's "
        "identity-continuity data — Iam, SOUL, self-model, relationship state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "state", "prefetch", "whoami", "reflect", "compact"],
            },
            "key": {"type": "string", "description": "Memory key (dot-path supported: 'user.name')"},
            "value": {"description": "Value to store (for 'write' and 'reflect')"},
            "op": {"type": "string", "enum": ["set", "delete", "merge", "append", "patch"],
                    "description": "Operation type (default: 'set')"},
            "query": {"type": "string", "description": "Search query (for 'prefetch')"},
            "reason": {"type": "string", "description": "Why this change was made (for 'reflect')"},
        },
        "required": ["action"],
    },
}


class OpenLLMMemoryProvider:
    """Hermes MemoryProvider for the Δ Capsule system.
    
    Wraps Capsule + Soul + Iam into a single provider that
    the Hermes MemoryManager can load.
    """
    
    def __init__(self, store_dir: str = "~/.openllm/capsule"):
        self._store_dir = Path(store_dir).expanduser().resolve()
        self._capsule: Capsule | None = None
        self._soul: Soul | None = None
        self._iam: Iam | None = None
        self._initialized = False
        self._session_id = ""
    
    @property
    def name(self) -> str:
        return "openllm-memory"
    
    def is_available(self) -> bool:
        """Always available — fully local, no deps besides stdlib + pyyaml"""
        return True
    
    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._capsule = Capsule.open(str(self._store_dir))
        self._soul = self._load_or_create_soul()
        self._iam = Iam(str(self._store_dir / "iam"), self._capsule)
        self._initialized = True
        logger.info("openllm-memory initialized: %s soul=%s %dΔ",
                     self._store_dir,
                     self._soul.get("name", "?"),
                     self._capsule.delta_count())
    
    def _load_or_create_soul(self) -> Soul:
        """加载已有的SOUL，或创建一个默认的"""
        soul_path = self._store_dir / "soul.json"
        if soul_path.exists():
            return Soul.load(str(soul_path))
        
        # 创建默认SOUL
        soul = Soul.create({
            "name": "Hermes Agent",
            "type": "agent",
            "model": "unknown",
            "description": "Memory-aware AI agent with identity continuity",
        })
        soul.save(str(soul_path))
        return soul
    
    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        
        iam_data = self._iam.all()
        iam_text = ""
        if iam_data:
            iam_text = "Self-narrative: " + json.dumps(iam_data, ensure_ascii=False)
        
        return (
            "You have a persistent identity that survives across sessions. "
            "Your SOUL (identity) and Iam (self-narrative) persist between conversations. "
            "Use the `capsule` tool to write, read, and update your memory.\n"
            + (f"\n{iam_text}" if iam_text else "")
            + f"\nΔ count: {self._capsule.delta_count()}"
        )
    
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._initialized:
            return ""
        return self._capsule.prefetch(query) or ""
    
    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        # Auto-save conversation turns to capsule
        if self._initialized:
            self._capsule.write(
                f"conversation/{session_id or self._session_id}/turn",
                {"user": user_content[:200], "assistant": assistant_content[:200]},
                op="append",
                session_id=session_id or self._session_id,
            )
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [CAPSULE_TOOL_SCHEMA]
    
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if tool_name != "capsule":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        action = args.get("action", "")
        
        try:
            if action == "write":
                key = args.get("key", "")
                value = args.get("value")
                op = args.get("op", "set")
                delta = self._capsule.write(
                    key, value, op=op,
                    session_id=self._session_id,
                )
                return json.dumps({"success": True, "delta_id": delta.delta_id})
            
            elif action == "read":
                key = args.get("key", "")
                value = self._capsule.get(key)
                return json.dumps({"key": key, "value": value})
            
            elif action == "state":
                state = self._capsule.state()
                return json.dumps({"state": state, "delta_count": self._capsule.delta_count()})
            
            elif action == "prefetch":
                query = args.get("query", "")
                result = self._capsule.prefetch(query)
                return json.dumps({"result": result} if result else {"result": "", "note": "No relevant memories"})
            
            elif action == "whoami":
                return json.dumps({
                    "soul": self._soul.to_dict() if self._soul else {},
                    "iam": self._iam.all() if self._iam else {},
                    "soul_verified": self._soul.verify() if self._soul else False,
                })
            
            elif action == "reflect":
                value = args.get("value", {})
                reason = args.get("reason", "agent self-reflection")
                if self._iam:
                    self._iam.update(value, reason=reason, session_id=self._session_id)
                return json.dumps({"success": True, "iam": self._iam.all() if self._iam else {}})
            
            elif action == "compact":
                count = self._capsule.compact()
                return json.dumps({"compacted": count})
            
            return json.dumps({"error": f"Unknown action: {action}"})
        
        except Exception as e:
            logger.error("capsule handle_tool_call failed: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})
    
    def shutdown(self) -> None:
        if self._capsule:
            count = self._capsule.compact()
            logger.info("openllm-memory shutdown: compacted %d deltas", count)
