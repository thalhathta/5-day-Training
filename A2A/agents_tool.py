from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

from mcp.shared.memory import create_connected_server_and_client_session
import campus_tools_mcp  # uses the FastMCP app from Lesson 2


def _build_schema_map(tools_meta: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: (t.get("inputSchema") or {}) for t in tools_meta}


def _extract_week(text: str) -> Optional[int]:
    m = re.search(r"\bweek\s*(\d+)\b", text, flags=re.I)
    return int(m.group(1)) if m else None


def _extract_course_code(text: str) -> Optional[str]:
    m = re.search(r"\b[A-Z]{3,6}\d{3,5}\b", text)
    return m.group(0) if m else None


def _extract_staff_name(text: str) -> Optional[str]:
    m = re.search(r"\bfor\s+((Dr|Prof|Ms|Mr)\.?\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,3})\b", text)
    return m.group(1).strip() if m else None


def _extract_room_token(text: str) -> Optional[str]:
    m = re.search(r"\bLT-\d+\b|\bCS-Lab-\d+\b|\bLIB-Meeting-\d+\b|\b[A-Za-z]{2,5}-[A-Za-z]+-\d+\b", text)
    return m.group(0) if m else None


class CampusInfoToolAgent:
    """Tool-using campus info agent: LLM selects an MCP tool and answers from tool results only.
    Uses an in-memory MCP session for workshop reliability.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

        self.tools_meta: List[Dict[str, Any]] = []
        self.schema_map: Dict[str, Dict[str, Any]] = {}

        self._session = None
        self._session_cm = None

        self._graph = self._build_graph()

    async def initialize(self) -> "CampusInfoToolAgent":
        # In-memory MCP server+client session (no subprocess)
        self._session_cm = create_connected_server_and_client_session(
            campus_tools_mcp.mcp, raise_exceptions=True
        )
        self._session = await self._session_cm.__aenter__()

        tools_obj = await self._session.list_tools()
        self.tools_meta = [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in tools_obj.tools
        ]
        self.schema_map = _build_schema_map(self.tools_meta)
        return self

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None

    # ---------------- LangGraph nodes ----------------
    def _decide_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]

        tool_lines = []
        for t in self.tools_meta:
            name = t["name"]
            desc = t.get("description", "") or ""
            schema = self.schema_map.get(name, {})
            req = schema.get("required", []) or []
            props = list((schema.get("properties") or {}).keys())
            tool_lines.append(f"- {name}: {desc} | required={req} | keys={props}")
        tool_list_md = "\n".join(tool_lines)

        system = (
            "You are a campus information assistant for university faculty.\n"
            "You MUST choose exactly one tool from the available tools.\n"
            "Return ONLY valid JSON with keys: tool_name, arguments.\n"
            "IMPORTANT: argument keys MUST match schema keys exactly.\n"
            "No markdown. No extra keys.\n"
        )

        example = '{"tool_name":"find_room","arguments":{"building":"CS","room":"CS-Lab-2"}}'
        user = (
            f"AVAILABLE TOOLS (with schemas):\n{tool_list_md}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"Return JSON like: {example}"
        )

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        raw = resp.output_text.strip()
        try:
            obj = json.loads(raw)
            return {**state, "tool_call": obj, "decision_raw": raw, "error": None}
        except Exception:
            return {**state, "tool_call": None, "decision_raw": raw, "error": "Invalid JSON from decide_tool"}

    def _normalize_tool_call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tool_call = state.get("tool_call") or {}
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("arguments") or {}
        question = state.get("question", "")
        q_lower = question.lower()

        schema = self.schema_map.get(tool_name, {})
        required = set(schema.get("required", []) or [])

        if tool_name == "find_room":
            room_token = args.get("room") or _extract_room_token(question)
            building = args.get("building")
            if room_token and not building:
                building = room_token.split("-")[0].upper()
            if room_token:
                args = {"building": building or "", "room": room_token}

        elif tool_name == "list_contacts":
            if "it helpdesk" in q_lower or "helpdesk" in q_lower:
                args = {"contact_type": "it_helpdesk"}
            elif "library" in q_lower:
                args = {"contact_type": "library"}
            elif "academic office" in q_lower:
                args = {"contact_type": "academic_office"}

        elif tool_name == "find_timetable":
            cc = args.get("course_code") or _extract_course_code(question)
            wk = args.get("week")
            if wk is None:
                wk = _extract_week(question)
            new_args = {}
            if cc:
                new_args["course_code"] = cc
            if wk is not None:
                new_args["week"] = int(wk)
            args = new_args

        elif tool_name == "get_office_hours":
            sn = args.get("staff_name") or _extract_staff_name(question)
            if sn:
                args = {"staff_name": sn}

        elif tool_name == "find_staff":
            if "query" not in args or not args.get("query"):
                sn = _extract_staff_name(question)
                if sn:
                    args = {"query": sn.split()[-1]}

        missing = [k for k in required if not args.get(k)]
        if missing:
            return {**state, "tool_call": {"tool_name": tool_name, "arguments": args}, "arg_warning": f"Missing required args: {missing}"}
        return {**state, "tool_call": {"tool_name": tool_name, "arguments": args}}

    async def _call_tool(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Agent not initialized. Call await agent.initialize() first.")

        tool_call = state.get("tool_call")
        if not tool_call:
            return {**state, "tool_result": None}

        tool_name = tool_call["tool_name"]
        arguments = tool_call.get("arguments", {}) or {}

        result = await self._session.call_tool(tool_name, arguments)

        def _content_to_text(parts) -> str:
            if parts is None:
                return ""
            out = []
            for p in parts:
                out.append(p.text if hasattr(p, "text") else str(p))
            return "\n".join(out)

        tool_output = (
            result.structuredContent
            if result.structuredContent is not None
            else {"content_text": _content_to_text(result.content)}
        )

        return {**state, "tool_result": {"tool_name": tool_name, "arguments": arguments, "output": tool_output}}

    def _draft_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        tool_result = state.get("tool_result")
        warning = state.get("arg_warning")

        system = (
            "You are a campus information assistant for university faculty.\n"
            "STRICT RULES:\n"
            "1) Answer ONLY using TOOL_RESULT below.\n"
            "2) If TOOL_RESULT is empty or contains an error, say you cannot find it and suggest what to try next.\n"
            "3) Output exactly in this format:\n"
            "Answer:\n"
            "- ...\n\n"
            "Evidence:\n"
            "- Tool: <tool_name>\n"
            "- Key fields: ...\n\n"
            "Actions / Next steps:\n"
            "- ...\n"
        )

        tool_json = json.dumps(tool_result, indent=2, ensure_ascii=False)
        user = f"QUESTION:\n{question}\n\nTOOL_RESULT:\n{tool_json}"
        if warning:
            user += f"\n\nARG_WARNING:\n{warning}"

        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        return {**state, "final_answer": resp.output_text}

    def _build_graph(self):
        g = StateGraph(dict)
        g.add_node("decide_tool", self._decide_tool)
        g.add_node("normalize_tool_call", self._normalize_tool_call)
        g.add_node("call_tool", self._call_tool)
        g.add_node("draft_answer", self._draft_answer)

        g.set_entry_point("decide_tool")
        g.add_edge("decide_tool", "normalize_tool_call")
        g.add_edge("normalize_tool_call", "call_tool")
        g.add_edge("call_tool", "draft_answer")
        g.add_edge("draft_answer", END)
        return g.compile()

    async def answer_query(self, question: str) -> str:
        out = await self._graph.ainvoke({"question": question})
        return out["final_answer"]
