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

from agents import CoursePolicyAgent


class CoursePolicyAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = CoursePolicyAgent(
            pdf_path=os.environ.get(
                "COURSE_POLICY_PDF", "/home/robomy/Desktop/THALHATH/5-day-Training/A2A/SENG3200_Course_Policy_Handbook.pdf"
            )
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        prompt = context.get_user_input()
        response = self.agent.answer_query(prompt)
        await event_queue.enqueue_event(new_agent_text_message(response))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main() -> None:
    load_dotenv()
    host = os.environ.get("AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("COURSE_POLICY_AGENT_PORT", "9999"))

    skill = AgentSkill(
        id="course_policy_qna",
        name="Course Policy Q&A",
        description="Answers questions grounded in the course policy handbook (late penalties, extensions, integrity, AI tool rules, appeals).",
        tags=["course", "policy", "rubric", "academic integrity"],
        examples=[
            "What is the late submission penalty after 2 days?",
            "Are AI tools permitted for the Reflection Log?",
            "How do students appeal a marking decision?",
        ],
    )

    agent_card = AgentCard(
        name="CoursePolicyAgent",
        description="A document-grounded assistant for course policy questions. Answers only using the provided handbook.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=CoursePolicyAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"✅ Running CoursePolicyAgent at http://{host}:{port}/")
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()
