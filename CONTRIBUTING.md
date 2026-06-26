# Contributing to Cairn

Cairn is a reference build maintained by [Dust AI Labs](https://github.com/dustailabs).
Issues and PRs are welcome — a few notes to keep changes easy to review:

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

All tests run fully offline — no `ANTHROPIC_API_KEY` or network access is
needed. The retrieval layer is tested directly, and the crew's output
parsing/citation-validation logic is tested as plain Python functions,
independent of `crewai`'s actual `kickoff()` call. One integration test
exercises the real `crewai` agent/task wiring with a stub LLM and is
skipped automatically if `crewai` isn't installed.

## Code style

Formatted and linted with [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
```

## Adding a new agent role

1. Add the role to `cairn/crew/agents.py`.
2. Add its task (with the right `context=[...]` chain) to `cairn/crew/tasks.py`.
3. If the role needs its own tool, add it in `cairn/crew/tools.py` following
   the `build_retrieval_tool` pattern — bind it to a tracer so its calls
   show up in the trace.
4. Update `ResearchCrew.run` to include the new agent/task in the crew.

## Swapping the embedder or vector backend

`cairn/retrieval/embeddings.py` defines the `Embedder` protocol. Implement
`embed(texts) -> np.ndarray` against any model/provider and pass it into
`HybridRetriever` — nothing else needs to change. The same goes for
`VectorIndex`: replace the brute-force cosine search with an ANN backend
(FAISS, pgvector, Qdrant) behind the same `search()` signature for larger
corpora.
