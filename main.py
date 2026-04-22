"""
main.py — CLI entry point for the AutoStream AI Agent.

Runs an interactive terminal conversation that:
  - Maintains state across turns (memory)
  - Routes through the LangGraph workflow
  - Prints agent responses cleanly

Usage:
    python main.py

Environment:
    Groq_API_KEY must be set (see README).
"""

import os
import sys
from typing import Optional

from agent import get_graph, AgentState
from agent.state import Message, LeadInfo


# ---------------------------------------------------------------------------
# ANSI colour helpers for readable terminal output
# ---------------------------------------------------------------------------

def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m"

def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m"

def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"

def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------

def _initial_state() -> AgentState:
    """Return a blank state for a new conversation."""
    return AgentState(
        messages=[],
        intent="unknown",
        stage="idle",
        lead=LeadInfo(name=None, email=None, platform=None),
        pending_field=None,
        response="",
    )


# ---------------------------------------------------------------------------
# Pretty-print the agent's response (handles **bold** markdown minimally)
# ---------------------------------------------------------------------------

def _print_agent(text: str) -> None:
    # Replace **bold** with terminal bold for nicer UX
    import re
    formatted = re.sub(r"\*\*(.+?)\*\*", lambda m: _bold(m.group(1)), text)
    print(f"\n{_cyan(_bold('AutoStream AI'))}:\n{formatted}\n")


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run():
    """Start and manage the interactive CLI session."""

    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print(_yellow(
                    "⚠️  GROQ_API_KEY is not set.\n"
                    "For Windows (CMD): set GROQ_API_KEY=your-key-here\n"
                    "For PowerShell: $env:GROQ_API_KEY='your-key-here'\n"
                    "For Mac/Linux: export GROQ_API_KEY='your-key-here'\n"))
        sys.exit(1)

    graph = get_graph()
    state: AgentState = _initial_state()

    print(_bold("\n" + "="*60))
    print(_bold("  🎬  AutoStream AI Agent — Social-to-Lead Workflow"))
    print(_bold("="*60))
    print(_dim("  Type 'quit' or 'exit' to end the session.\n"))

    turn_count = 0

    while True:
        # --- Get user input ---
        try:
            user_input = input(_bold("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print(_green("\nThanks for chatting! See you on AutoStream. 🎬\n"))
            break

        turn_count += 1

        # --- Append user message to history ---
        user_msg: Message = {"role": "user", "content": user_input}
        state = AgentState(**{**state, "messages": state["messages"] + [user_msg]})

        # --- Run through the graph ---
        try:
            state = graph.invoke(state)
        except Exception as e:
            print(_yellow(f"\n⚠️  Agent error: {e}\n"))
            continue

        # --- Display response ---
        _print_agent(state.get("response", ""))

        # --- Debug: show state summary (comment out in production) ---
        print(_dim(
            f"  [State] stage={state.get('stage')} | "
            f"intent={state.get('intent')} | "
            f"lead={state.get('lead')}"
        ))

        # --- Graceful end after capture ---
        if state.get("stage") == "captured":
            print(_green("\n✅ Lead capture complete. Starting a new session would reset the conversation.\n"))
            another = input(_dim("Would you like to continue chatting? (yes/no): ")).strip().lower()
            if another not in {"yes", "y"}:
                print(_green("\nThank you! Have an amazing day. 🚀\n"))
                break
            # Reset stage to 'answering' so follow-up questions work
            state = AgentState(**{**state, "stage": "answering"})


if __name__ == "__main__":
    run()
