import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import re

import google.generativeai as genai
from rich.console import Console

PROMPT_TEMPLATE = """You are an expert CI and release engineering assistant.
You can: analyze failures, reason about logs, and answer general information requests by
invoking MCP tools (whose outputs are provided to you) and synthesizing helpful answers.

INSTRUCTIONS:
- Read the user's request and the tool output (if any).
- If there is a failure or error content, analyze cause and provide fixes.
- Otherwise, produce a complete and direct answer for the user's information request.
- Be precise, avoid speculation, and prefer concrete, actionable guidance.

OUTPUT FORMAT:
- If analyzing errors/failures:
  Summary: <1–2 sentence high-level failure summary>
  Likely Cause(s): <concise cause(s) based on evidence>
  Suggested Fixes: <up to 3 actionable steps>
  Evidence: <short quotes or key lines from the tool output>

- Otherwise (information request):
  Answer: <direct, precise answer>
  Details: <supporting details, data or steps>
  Notes: <optional clarifications, caveats>

User Request:
{user_query}

Tool Output:
{tool_text}

Your response:"""

logger = logging.getLogger(__name__)
console = Console()


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
        prompt = PROMPT_TEMPLATE.format(user_query=user_query, tool_text=tool_text)
        try:
            def _generate():
                return self.model.generate_content(prompt)

            resp = await asyncio.get_event_loop().run_in_executor(None, _generate)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return ""

    async def analyze_failure_log(self, log_text: str) -> str:
        # Backward compatibility: route to the generalized generator
        return await self.generate_answer("Analyze the following error log and provide causes and fixes.", log_text)

    async def answer_query(
        self,
        query: str,
        connected_clients: List[Tuple[Any, str, List[Dict[str, Any]]]],
    ):
        # Simple heuristic kept for compatibility if needed in future
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
        """Plan which server/tool to call and with what arguments.

        Returns a dict like: {"server": str, "tool": str, "args": {..}} or {"server": null} if none.
        """
        system_prompt = (
            "You are a tool planner. Given the user request and available MCP servers and their tools, "
            "choose EXACTLY ONE best tool to answer the request, and construct its arguments from the user text. "
            "If required inputs are missing, infer them from the query when possible. If truly missing, set them to null.\n\n"
            "Return ONLY a strict JSON object with keys: server, tool, args. No markdown, no text outside JSON."
        )

        inventory = []
        for server in tools_by_server:
            server_name = server.get("server") or ""
            compact_tools = []
            for t in server.get("tools", []):
                name = t.get("name", "")
                schema = t.get("inputSchema") or {}
                props = {}
                required = []
                if isinstance(schema, dict):
                    required = schema.get("required") or []
                    if isinstance(schema.get("properties"), dict):
                        props = {k: (v.get("type") if isinstance(v, dict) else None) for k, v in schema["properties"].items()}
                compact_tools.append({"name": name, "required": required, "properties": props})
            inventory.append({"server": server_name, "tools": compact_tools})

        plan_prompt = (
            f"User Request:\n{query}\n\n"
            f"Available Servers and Tools (JSON):\n{json.dumps(inventory, ensure_ascii=False)}\n\n"
            "Respond with JSON only, e.g.: {\"server\": \"My Server\", \"tool\": \"analyze_job\", \"args\": {\"job_id\": \"...\"}}"
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