import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from rich.console import Console

PROMPT_TEMPLATE = """You are an expert CI and release engineering assistant.
You can: analyze failures, reason about logs, and answer general information requests by
using MCP tools (whose outputs are provided to you) and synthesizing helpful answers.

INTENT MODES (choose exactly one based on the user's request):
1) Information Retrieval / Listing (e.g., list/get/show/fetch/find/is/are/latest/open/unresolved/status/count/details)
   - Provide a direct, concise answer based on the tool output.
   - For lists, present a readable list of items (include key identifiers and short summaries/status).
   - Do NOT include failure analysis or fixes in this mode.
   - If no data is available, say so clearly (e.g., "No open tickets found").

2) Failure / Error Analysis (e.g., analyze/diagnose/debug/why failed/error/root cause)
   - Provide cause analysis and actionable fixes using the format below.

STRICT RULES:
- Do NOT speculate beyond the tool output.
- Do NOT convert a listing/lookup request into an error analysis.
- Keep answers precise and aligned with the user’s intent.

OUTPUT FORMAT:
- If Information Retrieval / Listing:
  Answer: <direct, precise answer>
  Items:
  - <KEY or ID>: <short summary> (<status>)
  Details: <supporting details if helpful>
  Notes: <optional clarifications>

- If Failure / Error Analysis:
  Summary: <1–2 sentence high-level failure summary>
  Likely Cause(s): <concise cause(s) based on evidence>
  Suggested Fixes: <up to 3 actionable steps>
  Evidence: <short quotes or key lines from the tool output>

User Request:
{user_query}

Tool Output:
{tool_text}

{data_hint}
Your response:"""

logger = logging.getLogger(__name__)
console = Console()


def _extract_list_from_text(tool_text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON list payload from tool_text.

    Looks for top-level array or common keys: products, tickets, items, results, data.
    Returns {"key": str, "items": List[Dict|Any]} or None.
    """
    try:
        # Find the first JSON object/array in the text
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", tool_text)
        if not m:
            return None
        obj = json.loads(m.group(1))
        # Direct list
        if isinstance(obj, list):
            return {"key": "items", "items": obj}
        if isinstance(obj, dict):
            for key in ["products", "tickets", "items", "results", "data"]:
                val = obj.get(key)
                if isinstance(val, list):
                    return {"key": key, "items": val}
        return None
    except Exception:
        return None


@dataclass
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.0-flash-exp"


class GeminiAgent:
    def __init__(self, config: GeminiConfig):
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)
        self.gemini_config = config

    async def generate_answer(self, user_query: str, tool_text: str) -> str:
        data_hint = ""
        extracted = _extract_list_from_text(tool_text or "")
        if extracted and isinstance(extracted.get("items"), list):
            # Provide a compact JSON snapshot limited to first 20 items and common fields
            items = extracted["items"][:20]

            # Try to reduce noise by picking basic fields when dicts
            def _shrink(x: Any) -> Any:
                if isinstance(x, dict):
                    for k in [
                        "id",
                        "name",
                        "key",
                        "title",
                        "summary",
                        "status",
                        "state",
                        "version",
                        "product",
                    ]:
                        if k in x:
                            # keep common fields only
                            return {
                                kk: x[kk]
                                for kk in x
                                if kk
                                in {
                                    "id",
                                    "name",
                                    "key",
                                    "title",
                                    "summary",
                                    "status",
                                    "state",
                                    "version",
                                    "product",
                                }
                            }
                    return {k: v for k, v in list(x.items())[:5]}
                return x

            shrunk = [_shrink(i) for i in items]
            data_hint = (
                "DATA_JSON (for listing mode):\n"
                + json.dumps(
                    {
                        "key": extracted["key"],
                        "count": len(extracted["items"]),
                        "items": shrunk,
                    },
                    ensure_ascii=False,
                )
                + "\nGUIDANCE: Use DATA_JSON to produce a list; avoid generic success messages."
            )

        prompt = PROMPT_TEMPLATE.format(
            user_query=user_query, tool_text=tool_text, data_hint=data_hint
        )
        try:

            def _generate():
                return self.model.generate_content(prompt)

            resp = await asyncio.get_event_loop().run_in_executor(None, _generate)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return ""

    async def analyze_failure_log(self, log_text: str) -> str:
        return await self.generate_answer(
            "Analyze the following error log and provide causes and fixes.", log_text
        )

    async def answer_query(
        self,
        query: str,
        connected_clients: List[Tuple[Any, str, List[Dict[str, Any]]]],
    ):
        relevant_text = query
        for client, server_name, tools in connected_clients:
            try:
                for tool in tools:
                    name = tool.get("name", "").lower()
                    if any(k in name for k in ["analy", "log", "fail"]):
                        pass
            except Exception:
                continue
        return await self.generate_answer(query, relevant_text)

    async def plan_tool_call(
        self,
        query: str,
        tools_by_server: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a tool planner. Given the user request and available MCP servers and their tools, "
            "choose EXACTLY ONE best tool to answer the request, and construct its arguments from the user text. "
            "CRITICAL RULES:\n"
            "1. Analyze the tool's required parameters and provide reasonable values based on the user query\n"
            "2. For missing required parameters, infer sensible defaults from context:\n"
            "   - instance_type: default to 'upstream' unless user specifies 'downstream'\n"
            "   - branch_name: default to 'main' unless user specifies another branch\n"
            "   - status: infer from query ('failed', 'success', 'running', etc.)\n"
            "   - pipeline_id: only use if explicitly mentioned or if this is a follow-up query\n"
            "3. NEVER use null, None, or empty string for required parameters\n"
            "4. If a query requires multiple steps (e.g., 'get errors for latest failed pipeline'), "
            "   choose the FIRST logical step (e.g., get_latest_pipeline with status='failed')\n\n"
            "Return ONLY a strict JSON object with keys: server, tool, args. No markdown."
        )
        inventory = []
        for server in tools_by_server:
            server_name = server.get("server") or ""
            compact_tools = []
            for t in server.get("tools", []):
                name = t.get("name", "")
                description = t.get("description", "")
                schema = t.get("inputSchema") or {}
                props = {}
                required = []
                if isinstance(schema, dict):
                    required = schema.get("required") or []
                    if isinstance(schema.get("properties"), dict):
                        props = {}
                        for k, v in schema["properties"].items():
                            if isinstance(v, dict):
                                prop_info = {
                                    "type": v.get("type"),
                                    "description": v.get("description", ""),
                                }
                                if "enum" in v:
                                    prop_info["enum"] = v["enum"]
                                props[k] = prop_info
                            else:
                                props[k] = {"type": None}
                compact_tools.append(
                    {"name": name, "description": description, "required": required, "properties": props}
                )
            inventory.append({"server": server_name, "tools": compact_tools})
        plan_prompt = (
            f"User Request:\n{query}\n\n"
            f"Available Servers and Tools (JSON):\n{json.dumps(inventory, ensure_ascii=False, indent=2)}\n\n"
            "EXAMPLES of proper parameter inference:\n"
            '- "latest failed pipeline" → {"instance_type": "upstream", "status": "failed", "branch_name": "main"}\n'
            '- "pipeline errors" (requires pipeline_id) → choose get_latest_pipeline first\n'
            '- "downstream pipeline" → {"instance_type": "downstream"}\n'
            '- "pipeline on branch dev" → {"branch_name": "dev"}\n\n'
            'Respond with JSON only, e.g.: {"server": "Pipelines MCP Server", "tool": "get_latest_pipeline", "args": {"instance_type": "upstream", "status": "failed"}}'
        )

        def _generate_plan():
            return self.model.generate_content([system_prompt, plan_prompt])

        try:
            resp = await asyncio.get_event_loop().run_in_executor(None, _generate_plan)
            text = getattr(resp, "text", "") or ""
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return {"server": None}
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict):
                return {"server": None}
            srv = obj.get("server")
            tool = obj.get("tool")
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            if not srv or not tool:
                return {"server": None}
            return {"server": str(srv), "tool": str(tool), "args": args}
        except Exception as e:
            logger.error(f"Gemini planning failed: {e}")
            return {"server": None}
