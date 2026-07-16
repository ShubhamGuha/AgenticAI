"""Healthcare research assistant for MediSyn Labs using AutoGen and Gemini.

This script provides a Streamlit UI for medical and clinical research queries,
including summarization, treatment comparison, and stateful memory across
interactions. It reads the Gemini API key from a project-local .env file.
"""

from __future__ import annotations

import csv
import importlib
import io
import os
import re
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from autogen import AssistantAgent, UserProxyAgent
from docx import Document
from dotenv import load_dotenv
from fpdf import FPDF

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

# Runtime bootstrap: verify that required dependencies are available before the UI starts.


@dataclass
class AgentState:
    """Simple state container for healthcare research interactions."""

    researcher_id: str = "researcher_01"
    project_id: str = "project_01"
    disease_focus: str = "general"
    short_term_memory: List[Dict[str, Any]] = field(default_factory=list)
    long_term_memory: List[Dict[str, Any]] = field(default_factory=list)


def ensure_runtime_environment() -> None:
    """Ensure the script runs with the project virtual environment."""
    required_modules = {
        "streamlit": "streamlit",
        "dotenv": "python-dotenv",
        "docx": "python-docx",
        "fpdf": "fpdf",
        "autogen": "autogen",
        "google.genai": "google-genai",
        "vertexai": "vertexai",
    }

    missing = []
    for module_name, package_name in required_modules.items():
        try:
            importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            missing.append(package_name)

    if not missing:
        return

    venv_candidates = [
        ROOT_DIR / "autogen_hca_venv" / "Scripts" / "python.exe",
        ROOT_DIR / "autogen_venv" / "Scripts" / "python.exe",
        ROOT_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    for venv_python in venv_candidates:
        if venv_python.exists() and sys.executable != str(venv_python):
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:])
            return

    raise SystemExit(
        "Missing Python packages: " + ", ".join(missing) + ".\n"
        "Install them into the project virtual environment with:\n"
        "  .\\autogen_hca_venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


# Ensure the correct virtual environment is active and load project-local settings.
ensure_runtime_environment()
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Configure the Streamlit page shell before rendering UI elements.
st.set_page_config(page_title="MediSyn Healthcare Assistant", page_icon="🧬", layout="wide")

if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState()
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "research"


def get_gemini_settings() -> tuple[str, str]:
    """Read Gemini configuration from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    return api_key, model_name


def is_informational(query: str) -> bool:
    """Filter out greetings and vague inputs before storing them in memory."""
    text = query.strip().lower()
    if not text:
        return False
    if len(text.split()) < 2:
        return False
    if re.match(r"^(hi|hello|hey|thanks|thank you|good morning|good evening|bye|ok|okay)$", text):
        return False
    if any(token in text for token in ["what can you do", "who are you", "help me"]):
        return False
    return True


def update_state(state: AgentState, user_query: str, response: str, mode: str) -> None:
    """Update short-term and long-term memory while keeping the state compact."""
    if not is_informational(user_query):
        return

    entry = {
        "query": user_query,
        "response": response[:800],
        "mode": mode,
    }
    state.short_term_memory.append(entry)
    if len(state.short_term_memory) > 7:
        state.short_term_memory = state.short_term_memory[-7:]

    long_entry = {
        "researcher_id": state.researcher_id,
        "project_id": state.project_id,
        "disease_focus": state.disease_focus,
        "mode": mode,
        "summary": response[:1200],
    }
    state.long_term_memory.append(long_entry)
    if len(state.long_term_memory) > 12:
        state.long_term_memory = state.long_term_memory[-12:]


def build_prompt(user_query: str, state: AgentState, mode: str) -> str:
    """Compose a task-specific prompt using short- and long-term memory."""
    short_memory_text = "\n".join(
        f"- {item['query']} -> {item['response'][:220]}"
        for item in state.short_term_memory[-5:]
    ) or "- No recent queries yet."

    long_memory_text = "\n".join(
        f"- {item['mode']} / {item['disease_focus']} / {item['summary'][:160]}"
        for item in state.long_term_memory[-4:]
    ) or "- No saved summaries yet."

    if mode == "summarize":
        instruction = (
            "Create a concise evidence-style summary for healthcare researchers. "
            "Use bullet points and highlight key findings, caveats, and next steps."
        )
    elif mode == "compare":
        instruction = (
            "Compare treatment options, therapies, or interventions for the request. "
            "Mention differences in efficacy, population, safety, and notable evidence quality."
        )
    else:
        instruction = (
            "Answer this medical or clinical research question clearly and safely. "
            "If the request is sensitive, mention uncertainty and suggest expert review."
        )

    return (
        f"{instruction}\n\n"
        f"Researcher ID: {state.researcher_id}\n"
        f"Project ID: {state.project_id}\n"
        f"Disease focus: {state.disease_focus or 'general'}\n\n"
        f"Short-term memory (most recent):\n{short_memory_text}\n\n"
        f"Long-term memory:\n{long_memory_text}\n\n"
        f"User query:\n{user_query}"
    )


def build_autogen_agents() -> tuple[AssistantAgent, UserProxyAgent]:
    """Create the AutoGen agents used for the healthcare workflow."""
    api_key, model_name = get_gemini_settings()
    if not api_key:
        raise RuntimeError("Add GOOGLE_API_KEY to the project .env file before running the app.")

    config_list = [{"model": model_name, "api_key": api_key, "api_type": "google"}]

    assistant = AssistantAgent(
        name="medical_assistant",
        llm_config={"config_list": config_list, "seed": 42},
        max_consecutive_auto_reply=1,
        system_message=(
            "You are a healthcare research assistant for MediSyn Labs. Respond clearly, "
            "use evidence-based reasoning, summarize findings succinctly, and flag uncertainty."
        ),
    )
    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"work_dir": "coding", "use_docker": False},
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    )
    return assistant, user_proxy


def ask_with_autogen(prompt: str) -> str:
    """Run a prompt through the AutoGen healthcare assistant."""
    assistant, user_proxy = build_autogen_agents()
    collected_messages: List[str] = []

    def capture_reply(self: Any, message: Any, sender: Any, request_reply: Any, silent: bool) -> None:
        if getattr(sender, "name", None) == "medical_assistant":
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            if content and content not in collected_messages:
                collected_messages.append(content)

    user_proxy.receive = capture_reply.__get__(user_proxy)
    user_proxy.initiate_chat(assistant, message=prompt)
    return "\n\n".join(collected_messages).strip() or "No response received."


# Main UI layout: sidebar metadata, query input, mode selection, and action buttons.
st.title("MediSyn Labs Healthcare Research Assistant")
st.caption("Research summaries, treatment comparisons, and clinician-facing memory support using AutoGen + Gemini")

with st.sidebar:
    st.header("Research metadata")
    st.session_state.agent_state.researcher_id = st.text_input(
        "Researcher ID",
        value=st.session_state.agent_state.researcher_id,
        key="researcher_id",
    )
    st.session_state.agent_state.project_id = st.text_input(
        "Project ID",
        value=st.session_state.agent_state.project_id,
        key="project_id",
    )
    st.session_state.agent_state.disease_focus = st.text_input(
        "Disease focus",
        value=st.session_state.agent_state.disease_focus,
        key="disease_focus",
    )
    st.info("The agent keeps up to 7 short-term interactions and 12 long-term summaries in memory.")

query = st.text_area("Enter a healthcare question or literature request", height=120, key="query_input")
mode = st.radio(
    "Task mode",
    ["research", "summarize", "compare"],
    format_func=lambda value: {
        "research": "Research answer",
        "summarize": "Summarize findings",
        "compare": "Compare treatments",
    }[value],
    horizontal=True,
)
st.session_state.selected_mode = mode

col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("Run agent", use_container_width=True)
with col2:
    approve_clicked = st.button("Request human approval", use_container_width=True)

if run_clicked:
    if not query.strip():
        st.warning("Please enter a healthcare query before running the agent.")
    else:
        with st.spinner("Generating response..."):
            try:
                prompt = build_prompt(query, st.session_state.agent_state, mode)
                response = ask_with_autogen(prompt)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Request failed: {exc}")
            else:
                st.session_state.assistant_messages = [response]
                update_state(st.session_state.agent_state, query, response, mode)
                st.success("Response generated successfully.")

if approve_clicked:
    st.info("Approval prompt: Review the generated summary before sharing it with researchers.")

if st.session_state.assistant_messages:
    st.markdown("### Assistant response")
    st.markdown(st.session_state.assistant_messages[-1])

with st.expander("Short-term memory"):
    if st.session_state.agent_state.short_term_memory:
        for item in st.session_state.agent_state.short_term_memory:
            st.write(f"- Query: {item['query']}")
            st.caption(f"Mode: {item['mode']} | Response: {item['response'][:180]}")
    else:
        st.info("No short-term memory yet.")

with st.expander("Long-term memory"):
    if st.session_state.agent_state.long_term_memory:
        for item in st.session_state.agent_state.long_term_memory:
            st.write(f"- {item['mode']} | {item['disease_focus']}")
            st.caption(item['summary'][:280])
    else:
        st.info("No long-term memory yet.")

# Export generated responses in common researcher-friendly formats when an answer exists.
if st.session_state.assistant_messages:
    response_text = st.session_state.assistant_messages[-1]
    doc = Document()
    doc.add_paragraph(response_text)
    word_buffer = BytesIO()
    doc.save(word_buffer)
    word_buffer.seek(0)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Assistant Response"])
    writer.writerow([response_text])
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    csv_buffer = BytesIO(csv_bytes)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=response_text)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_buffer = BytesIO(pdf_bytes)

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            label="Download Word",
            data=word_buffer,
            file_name="healthcare_summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with export_col2:
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name="healthcare_summary.csv",
            mime="text/csv",
        )
    with export_col3:
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="healthcare_summary.pdf",
            mime="application/pdf",
        )
