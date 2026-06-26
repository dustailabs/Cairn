import pytest

crewai = pytest.importorskip("crewai")

from cairn.crew.agents import build_agents  # noqa: E402
from cairn.crew.tasks import build_tasks  # noqa: E402
from cairn.crew.tools import build_retrieval_tool  # noqa: E402
from cairn.retrieval import DocumentStore, HashingEmbedder, HybridRetriever  # noqa: E402
from cairn.trace import Tracer  # noqa: E402


class _StubLLM(crewai.BaseLLM):
    """Minimal BaseLLM subclass so Agent's pydantic validation accepts it,
    without ever making a real call. Not exercised — these tests only
    confirm the agent/task/tool wiring is structurally correct."""

    model: str = "stub/stub-model"

    def __init__(self) -> None:
        super().__init__(model="stub/stub-model")

    def call(self, *args, **kwargs):  # pragma: no cover - not exercised
        return "stubbed response"


def test_build_agents_and_tasks_wires_without_calling_a_real_model():
    """Confirms the crewai objects construct correctly with our roles/tools/
    context chain wired up — without ever calling crew.kickoff() (and
    therefore without needing an API key or network access)."""
    store = DocumentStore()
    store.add_document(text="Kafka brokers replicate partitions.", source="kafka-doc")
    retriever = HybridRetriever(store.all_chunks(), HashingEmbedder())

    tool = build_retrieval_tool(retriever, Tracer())
    agents = build_agents(_StubLLM(), tool)

    assert set(agents) == {"planner", "researcher", "critic"}
    assert tool in agents["researcher"].tools

    tasks = build_tasks(agents, "How does Kafka replicate partitions?")
    assert len(tasks) == 3
    assert tasks[1].context == [tasks[0]]
    assert tasks[2].context == [tasks[1]]
