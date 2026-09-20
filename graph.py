"""LangGraph entry point for local graph tooling and Studio."""

from app import build_graph


# LangGraph tooling discovers this compiled graph object.
graph = build_graph()
