"""Role definitions for Cairn's three-agent crew.

Planner -> Researcher -> Critic, run as a sequential CrewAI process:

- **Planner** decomposes the user's question into focused sub-questions.
- **Researcher** calls the retrieval tool for each sub-question and
  collects grounded passages — it never answers from memory.
- **Critic** synthesizes a final answer using *only* what the Researcher
  retrieved, and must cite a chunk id for every claim.
"""

from __future__ import annotations


def build_agents(llm, retrieval_tool):
    """Build the Planner, Researcher, and Critic ``crewai.Agent`` objects."""
    from crewai import Agent  # type: ignore

    planner = Agent(
        role="Query Planner",
        goal="Break the user's question into the minimum set of focused sub-questions needed to answer it fully.",
        backstory=(
            "You are a meticulous research lead. You never answer questions yourself — "
            "you only decide what needs to be looked up, and in what order."
        ),
        llm=llm,
        verbose=False,
    )

    researcher = Agent(
        role="Researcher",
        goal="Retrieve grounded passages for every sub-question using the retrieval tool, and never answer from memory.",
        backstory=(
            "You are a careful research assistant. You only know what the retrieval tool "
            "returns to you. If the tool finds nothing relevant, you say so explicitly "
            "instead of filling the gap with prior knowledge."
        ),
        tools=[retrieval_tool],
        llm=llm,
        verbose=False,
    )

    critic = Agent(
        role="Critic",
        goal=(
            "Write the final answer using only the Researcher's retrieved passages, "
            "citing the chunk id of every passage you rely on, and flagging anything "
            "the passages don't support."
        ),
        backstory=(
            "You are a skeptical editor. You reject any claim in a draft answer that "
            "isn't directly backed by a cited passage, and you would rather say "
            "'the knowledge base doesn't cover this' than let an unsupported claim "
            "through."
        ),
        llm=llm,
        verbose=False,
    )

    return {"planner": planner, "researcher": researcher, "critic": critic}
