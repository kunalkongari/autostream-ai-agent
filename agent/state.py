"""
state.py — Defines the AgentState TypedDict used across all LangGraph nodes.

The state object is the single source of truth that flows through every node
in the graph. LangGraph passes it by value between nodes, so each node must
return a (possibly mutated) copy.
"""

from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
import operator


# ---------------------------------------------------------------------------
# Message type (lightweight — avoids importing langchain_core just for this)
# ---------------------------------------------------------------------------

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


# ---------------------------------------------------------------------------
# Lead collection progress
# ---------------------------------------------------------------------------

class LeadInfo(TypedDict):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


# ---------------------------------------------------------------------------
# Main agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Full conversation history (used for memory across turns)
    messages: Annotated[List[Message], operator.add]

    # Latest classified intent for the current turn
    intent: Literal["greeting", "product_inquiry", "high_intent", "unknown"]

    # Current stage in the workflow FSM
    # Stages:
    #   idle          → waiting for first message
    #   answering     → RAG-based Q&A mode
    #   collecting    → gathering lead details (name/email/platform)
    #   captured      → lead successfully saved, conversation can end
    stage: Literal["idle", "answering", "collecting", "captured"]

    # Collected lead fields (populated incrementally during "collecting" stage)
    lead: LeadInfo

    # Which field to ask for next during lead collection
    pending_field: Optional[Literal["name", "email", "platform"]]

    # The assistant's response to be returned for this turn
    response: str
