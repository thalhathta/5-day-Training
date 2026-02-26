from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


class WebResearchAgent:
    """Live web-search research agent using OpenAI's built-in web_search tool."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

        self.system_rules = """You are a research assistant for university faculty.

STRICT RULES:
1) Use web_search to find up-to-date information.
2) Your final output MUST include a Sources section with 3–6 sources.
3) Each source must include title and URL (if available from tool).
4) Do not invent citations. If sources are not available, say so explicitly.
5) Output exactly in this format:

Answer:
- ...

Sources:
- <Title> — <URL>
- ...

Notes:
- ...
"""

    def _extract_web_sources(self, resp: Any) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        output = getattr(resp, "output", None) or []
        for item in output:
            t = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if t == "web_search_call":
                action = getattr(item, "action", None) or (item.get("action") if isinstance(item, dict) else None)
                if action is None:
                    continue
                s = getattr(action, "sources", None) or (action.get("sources") if isinstance(action, dict) else None)
                if s:
                    for src in s:
                        if isinstance(src, dict):
                            sources.append(src)
                        else:
                            sources.append(getattr(src, "model_dump", lambda: {"title": str(src)})())
        return sources

    def answer_query(self, question: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
            input=[
                {"role": "system", "content": self.system_rules},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )

        text = (resp.output_text or "").strip()
        sources = self._extract_web_sources(resp)

        if "Sources:" not in text and sources:
            lines = [text, "", "Sources:"]
            for s in sources[:6]:
                title = s.get("title") or s.get("name") or "Source"
                url = s.get("url") or s.get("link") or ""
                lines.append(f"- {title}" + (f" — {url}" if url else ""))
            lines += ["", "Notes:", "- (Auto-added sources from tool metadata.)"]
            text = "\n".join(lines)

        return text
