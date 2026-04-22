"""
graph.py — Builds and compiles the LangGraph StateGraph.

Graph topology:
                           ┌──────────────┐
                           │  START (user) │
                           └──────┬───────┘
                                  │
                        ┌─────────▼──────────┐
                        │  classify_intent   │
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │     router         │ (conditional edge)
                        └──┬────────┬────────┘
                           │        │         │            │
                    greeting  product_  high_intent  collecting/
                              inquiry              capturing
                           │        │         │            │
                     ┌─────▼──┐ ┌──▼──┐  ┌───▼────┐  ┌───▼───────┐
                     │ greet  │ │ rag │  │collect │  │  capture  │
                     └────────┘ └─────┘  └────────┘  └───────────┘
                           │        │         │            │
                           └────────┴─────────┴────────────┘
                                          │
                                        END
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_intent_node,
    greet_node,
    rag_node,
    collect_lead_node,
    capture_lead_node,
)


# ---------------------------------------------------------------------------
# Routing logic (conditional edge)
# ---------------------------------------------------------------------------

def router(state: AgentState) -> str:
    """
    Determines which node to execute based on current state.

    Priority:
      1. If we're mid-collection → keep collecting (or capture if done)
      2. If we're already captured → go to rag (answer any follow-up)
      3. Otherwise route based on intent
    """
    stage = state.get("stage", "idle")
    intent = state.get("intent", "unknown")

    # Mid-collection: keep collecting regardless of intent
    if stage == "collecting":
        return "collect_lead"

    # All fields collected — time to capture
    if stage == "capturing":
        return "capture_lead"

    # Already captured — just answer any follow-ups via RAG
    if stage == "captured":
        return "rag"

    # Normal routing based on intent
    if intent == "greeting":
        return "greet"
    elif intent == "high_intent":
        return "collect_lead"
    elif intent == "product_inquiry":
        return "rag"
    else:
        # unknown intent → default to RAG (safe fallback)
        return "rag"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("greet", greet_node)
    graph.add_node("rag", rag_node)
    graph.add_node("collect_lead", collect_lead_node)
    graph.add_node("capture_lead", capture_lead_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional edge from classify_intent → router decides next node
    graph.add_conditional_edges(
        "classify_intent",
        router,
        {
            "greet": "greet",
            "rag": "rag",
            "collect_lead": "collect_lead",
            "capture_lead": "capture_lead",
        },
    )

    # All leaf nodes → END (each turn produces one response)
    graph.add_edge("greet", END)
    graph.add_edge("rag", END)
    graph.add_edge("collect_lead", END)
    graph.add_edge("capture_lead", END)

    return graph.compile()


# Singleton — compiled once and reused across all turns
_compiled_graph = None


def get_graph():
    """Return the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
