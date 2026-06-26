"""Task definitions chaining Planner -> Researcher -> Critic.

Each task's ``expected_output`` doubles as the contract that
:mod:`cairn.crew.parsing` validates against, the same way Concord's router
agent is told to emit a strict JSON shape and the orchestrator validates it
on the way out.
"""

from __future__ import annotations


def build_tasks(agents: dict, query: str):
    from crewai import Task  # type: ignore

    plan_task = Task(
        description=(
            f"User question: {query!r}\n\n"
            "List 1-4 focused sub-questions that, once answered, fully answer the "
            "user's question. Keep each sub-question self-contained and specific "
            "enough to search for directly."
        ),
        expected_output="A numbered list of sub-questions, nothing else.",
        agent=agents["planner"],
    )

    research_task = Task(
        description=(
            "For each sub-question from the plan, call the retrieval tool and collect "
            "the passages it returns. Do not answer the sub-questions yourself — only "
            "gather and forward what the tool finds, including its chunk ids."
        ),
        expected_output=(
            "All retrieved passages, each clearly labeled with its chunk id and source, "
            "grouped by the sub-question they answer."
        ),
        agent=agents["researcher"],
        context=[plan_task],
    )

    critic_task = Task(
        description=(
            f"Original user question: {query!r}\n\n"
            "Using only the passages the Researcher retrieved, write the final answer. "
            "Every factual claim must cite the exact chunk id it came from. If the "
            "passages don't fully answer the question, say so explicitly rather than "
            "filling the gap from general knowledge."
        ),
        expected_output=(
            "A single JSON object with keys: "
            '"answer" (string), "citations" (array of chunk id strings actually used), '
            'and "unresolved" (boolean — true if the passages did not fully answer the '
            "question). No text outside the JSON object."
        ),
        agent=agents["critic"],
        context=[research_task],
    )

    return [plan_task, research_task, critic_task]
