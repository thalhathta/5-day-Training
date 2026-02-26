import os
import uvicorn
from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

from agents_tool import CampusInfoToolAgent


class CampusInfoExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = None

    async def _ensure_init(self):
        if self.agent is None:
            self.agent = await CampusInfoToolAgent().initialize()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._ensure_init()
        prompt = context.get_user_input()
        response = await self.agent.answer_query(prompt)
        await event_queue.enqueue_event(new_agent_text_message(response))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main() -> None:
    load_dotenv()
    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CAMPUS_INFO_AGENT_PORT", "9998"))

    skill = AgentSkill(
        id="campus_info_tools",
        name="Campus Information Tools",
        description="Finds staff contacts, office hours, timetables, rooms, and helpdesk contacts using MCP tools.",
        tags=["campus", "staff", "timetable", "room", "contacts", "mcp"],
        examples=[
            "Find staff member Amina and give her email and office.",
            "What are the office hours for Dr Amina Rahman?",
            "Show timetable for SENG3200 week 3.",
            "Where is CS-Lab-2 and what facilities does it have?",
            "Give IT helpdesk contact details."
        ],
    )

    agent_card = AgentCard(
        name="CampusInfoToolAgent",
        description="A tool-using campus assistant. It chooses MCP tools and answers only from tool outputs.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=CampusInfoExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    print(f"✅ Running CampusInfoToolAgent at http://{host}:{port}/")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
