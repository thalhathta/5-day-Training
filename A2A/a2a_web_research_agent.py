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

from agents_research import WebResearchAgent


class WebResearchExecutor(AgentExecutor):
    def __init__(self) -> None:
        model = os.environ.get("RESEARCH_MODEL", "gpt-4o-mini")
        self.agent = WebResearchAgent(model=model)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = context.get_user_input()
        response = self.agent.answer_query(prompt)
        await event_queue.enqueue_event(new_agent_text_message(response))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main() -> None:
    load_dotenv()
    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_RESEARCH_AGENT_PORT", "9997"))

    skill = AgentSkill(
        id="live_web_research",
        name="Live Web Research",
        description="Performs live web search and returns a concise answer with source citations.",
        tags=["research", "web_search", "citations", "faculty"],
        examples=[
            "What are common university policies for AI tool usage in assessments?",
            "Summarize recent guidance on academic integrity and generative AI.",
            "Find best practices for rubric design and cite sources.",
        ],
    )

    agent_card = AgentCard(
        name="WebResearchAgent",
        description="A live web-search research agent for university faculty. Always returns sources.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=WebResearchExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    print(f"✅ Running WebResearchAgent at http://{host}:{port}/")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
