"""
nodes.py — LangGraph node functions.

Each function takes an AgentState and returns a partial state dict
(LangGraph merges it automatically via the Annotated reducers).

Node responsibilities:
  classify_intent_node   → detect what the user wants
  route_node             → decide which node to run next (conditional edge)
  greet_node             → handle greeting intent
  rag_node               → answer product/pricing questions
  collect_lead_node      → gather name / email / platform
  capture_lead_node      → call tool and finalise
"""

import os
from langchain_groq import ChatGroq

from .state import AgentState, LeadInfo, Message
from .intent import classify_intent
from .rag import retrieve, get_company_intro
from .tools import execute_lead_capture, extract_email_from_text, extract_platform_from_text

# ---------------------------------------------------------------------------
# Anthropic client (reused across calls)
# ---------------------------------------------------------------------------

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def _llm(system: str, messages: list[Message]) -> str:
    """
    Groq LLM wrapper using LangChain.
    """

    # Convert conversation to string
    conversation = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in messages]
    )

    full_prompt = f"""
{system}

Conversation so far:
{conversation}

Assistant:
"""

    llm = get_llm()
    response = llm.invoke(full_prompt)

    # Safe return
    if hasattr(response, "content"):
        return response.content.strip()

    return str(response).strip()


# ---------------------------------------------------------------------------
# Node 1 — Intent Classification
# ---------------------------------------------------------------------------

def classify_intent_node(state: AgentState) -> dict:
    """
    Reads the last user message and updates `intent` in state.
    Does NOT generate a response — routing happens separately.
    """
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )
    intent = classify_intent(last_user_msg)
    # print(f"[DEBUG] Intent classified: {intent}")
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node 2 — Greeting
# ---------------------------------------------------------------------------

def greet_node(state: AgentState) -> dict:
    """Produce a warm welcome and brief product intro."""
    intro = get_company_intro()
    response = (
        "👋 Hey there! Welcome to **AutoStream** — your AI-powered video editing co-pilot.\n\n"
        f"{intro}\n\n"
        "Whether you're a YouTuber, Instagrammer, or TikTok creator, we've got a plan for you. "
        "Feel free to ask about pricing, features, or anything else. How can I help you today?"
    )
    assistant_msg: Message = {"role": "assistant", "content": response}
    return {
        "messages": [assistant_msg],
        "response": response,
        "stage": "answering",
    }


# ---------------------------------------------------------------------------
# Node 3 — RAG Response
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    """
    Retrieve relevant knowledge base context, then generate a grounded answer.
    The LLM is explicitly instructed NOT to add info beyond what's retrieved.
    """
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

    # Retrieve relevant context
    context = retrieve(last_user_msg)

    system_prompt = (
        "You are the AutoStream support assistant. "
        "Answer the user's question using ONLY the information provided in the CONTEXT block below. "
        "Do not invent features, prices, or policies that are not in the context. "
        "Be concise, friendly, and use bullet points where helpful. "
        "If the user's question isn't covered by the context, say so honestly.\n\n"
        f"CONTEXT:\n{context}"
    )

    # Pass full conversation for multi-turn coherence
    response = _llm(system_prompt, state["messages"])

    # Append a nudge to upgrade if not already in high-intent stage
    if state.get("stage") != "collecting":
        response += (
            "\n\n💡 Interested in getting started? Just say the word and I'll walk you through signing up!"
        )

    assistant_msg: Message = {"role": "assistant", "content": response}
    return {
        "messages": [assistant_msg],
        "response": response,
        "stage": "answering",
    }


# ---------------------------------------------------------------------------
# Node 4 — Lead Collection (incremental field gathering)
# ---------------------------------------------------------------------------

def collect_lead_node(state: AgentState) -> dict:
    """
    Collects name, email, and platform one field at a time.
    
    On each call:
      1. Try to extract the field we were waiting for from the last user message.
      2. Determine the next missing field.
      3. Ask for it.
    
    When all three fields are collected, transition to capture stage.
    """
    lead: LeadInfo = state.get("lead") or {"name": None, "email": None, "platform": None}
    pending = state.get("pending_field")
    stage = state.get("stage", "idle")

    # If we're transitioning from answering → collecting, greet the collection
    if stage != "collecting":
        response = (
            "🎉 Awesome! I'd love to get you set up. Let me just grab a few quick details.\n\n"
            "First up — **what's your name?**"
        )
        assistant_msg: Message = {"role": "assistant", "content": response}
        return {
            "messages": [assistant_msg],
            "response": response,
            "stage": "collecting",
            "pending_field": "name",
            "lead": lead,
        }

    # Extract the field we were waiting for from the last user message
    last_user_msg = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

    if pending == "name" and not lead.get("name"):
        # Accept the entire message as the name (simple heuristic)
        extracted = last_user_msg.strip().split("\n")[0]  # first line
        if len(extracted) >= 2:
            lead = {**lead, "name": extracted}

    elif pending == "email" and not lead.get("email"):
        extracted = extract_email_from_text(last_user_msg)
        if extracted:
            lead = {**lead, "email": extracted}

    elif pending == "platform" and not lead.get("platform"):
        extracted = extract_platform_from_text(last_user_msg)
        if extracted:
            lead = {**lead, "platform": extracted}

    # Determine next missing field
    next_field = None
    if not lead.get("name"):
        next_field = "name"
    elif not lead.get("email"):
        next_field = "email"
    elif not lead.get("platform"):
        next_field = "platform"

    # Build response
    if next_field == "name":
        response = "I didn't quite catch your name. Could you tell me your **full name**?"
    elif next_field == "email":
        response = f"Nice to meet you, {lead['name']}! 😊 What's the best **email address** to reach you at?"
    elif next_field == "platform":
        response = (
            f"Got it! Last one — which **creator platform** are you primarily on? "
            f"(e.g. YouTube, Instagram, TikTok, etc.)"
        )
    else:
        # All fields collected — transition to capture
        response = f"Perfect! Let me get everything set up for you right now... 🚀"
        assistant_msg: Message = {"role": "assistant", "content": response}
        return {
            "messages": [assistant_msg],
            "response": response,
            "stage": "capturing",   # signals graph to route to capture node
            "pending_field": None,
            "lead": lead,
        }

    assistant_msg: Message = {"role": "assistant", "content": response}
    return {
        "messages": [assistant_msg],
        "response": response,
        "stage": "collecting",
        "pending_field": next_field,
        "lead": lead,
    }


# ---------------------------------------------------------------------------
# Node 5 — Lead Capture (tool execution)
# ---------------------------------------------------------------------------

def capture_lead_node(state: AgentState) -> dict:
    """
    Validates all collected fields and calls the mock_lead_capture tool.
    Must only be reached after all three fields are present.
    """
    lead = state.get("lead") or {}
    success, message = execute_lead_capture(
        name=lead.get("name"),
        email=lead.get("email"),
        platform=lead.get("platform"),
    )

    if success:
        response = (
            f"✅ **You're all set, {lead.get('name')}!**\n\n"
            f"We've registered your interest in AutoStream's Pro plan. "
            f"Our team will reach out to you at **{lead.get('email')}** within 24 hours "
            f"with your onboarding details.\n\n"
            f"In the meantime, feel free to explore our website. "
            f"Can't wait to see what you create on {lead.get('platform')}! 🎬"
        )
        new_stage = "captured"
    else:
        # Validation failed — go back to collecting
        response = f"⚠️ Hmm, something doesn't look right: {message}\n\nCould you double-check and try again?"
        new_stage = "collecting"

    assistant_msg: Message = {"role": "assistant", "content": response}
    return {
        "messages": [assistant_msg],
        "response": response,
        "stage": new_stage,
    }
