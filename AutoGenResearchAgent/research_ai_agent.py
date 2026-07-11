"""Streamlit-based research assistant powered by Google Gemini.

This script provides a single-file entrypoint for running a research assistant UI.
Users can enter a topic, ask the model for an answer, generate subtopics, or
summarize the latest response. The app reads its API configuration from a local
.env file and exports the current output to Word, CSV, and PDF files.
"""

import csv
import io
import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from autogen import AssistantAgent, UserProxyAgent
from docx import Document
from dotenv import load_dotenv
from fpdf import FPDF
from google import genai

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"


def ensure_runtime_environment():
    """Ensure the app runs with the project virtual environment and required packages.

    This helper prevents import failures when the current Python interpreter is
    not the environment that contains Streamlit, AutoGen, or the document export
    libraries used by the demo.
    """
    required_modules = {
        "streamlit": "streamlit",
        "dotenv": "python-dotenv",
        "docx": "python-docx",
        "fpdf": "fpdf",
        "google.genai": "google-genai",
    }

    missing = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    if not missing:
        return

    venv_python = ROOT_DIR / "autogen_venv" / "Scripts" / "python.exe"
    if venv_python.exists() and sys.executable != str(venv_python):
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:])

    raise SystemExit(
        "Missing Python packages: " + ", ".join(missing) + ".\n"
        "Install them into the project virtual environment with:\n"
        "  .\\autogen_venv\\Scripts\\python.exe -m pip install -r requirements.txt"
    )


# Ensure the script is using the environment that contains the required packages.
ensure_runtime_environment()

# Load environment variables from the project-local .env file.
load_dotenv(dotenv_path=ENV_PATH, override=False)

# Configure the Streamlit page metadata before rendering any content.
st.set_page_config(page_title="Research AI Agent", page_icon="🧠", layout="wide")

# Initialize session state so the app can preserve the prompt and generated
# responses across reruns.
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""


def get_gemini_settings():
    """Read Gemini settings from environment variables.

    The app prefers GOOGLE_API_KEY and optionally MODEL_NAME or GEMINI_MODEL.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    return api_key, model_name


def build_autogen_agents():
    """Create the AutoGen agents used to demonstrate the research workflow."""
    api_key, model_name = get_gemini_settings()
    if not api_key:
        raise RuntimeError("Add GOOGLE_API_KEY to the project .env file before running the app.")

    config_list = [{"model": model_name, "api_key": api_key, "api_type": "google"}]

    assistant = AssistantAgent(
        name="assistant",
        llm_config={"config_list": config_list, "seed": 42},
        max_consecutive_auto_reply=1,
        system_message=(
            "You are a research assistant. Answer the user's request clearly and, when asked, "
            "create structured subtopics and concise summaries."
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
    """Route a prompt through the AutoGen assistant and user-proxy agents."""
    assistant, user_proxy = build_autogen_agents()
    collected_messages = []

    def capture_reply(self, message, sender, request_reply, silent):
        # Capture the assistant's reply from the AutoGen conversation so it can be
        # shown in the Streamlit UI and exported by the user.
        if getattr(sender, "name", None) == "assistant":
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            if content:
                collected_messages.append(content)

    # Override the user proxy's receive callback so the assistant reply can be
    # captured while the AutoGen conversation executes.
    user_proxy.receive = capture_reply.__get__(user_proxy)
    user_proxy.initiate_chat(assistant, message=prompt)
    return "\n\n".join(collected_messages).strip() or "No response received."


# Render the main header and intro text for the app.
st.title("Research AI Agent")
st.caption("Ask a broad topic, generate subtopics, and summarize the latest response.")

# Main input box for the user's research question or topic.
topic = st.text_input("Enter your research question or topic:", key="topic_input")

# Workflow buttons that trigger the three supported actions.
col1, col2, col3 = st.columns(3)
with col1:
    ask_clicked = st.button("Ask", use_container_width=True)
with col2:
    subtopics_clicked = st.button("Generate Subtopics", use_container_width=True)
with col3:
    summarise_clicked = st.button("Summarise", use_container_width=True)


# Handle the primary Ask action.
if ask_clicked:
    if not topic.strip():
        st.warning("Please enter a research question or topic.")
    else:
        prompt = topic.strip()
        st.session_state.last_prompt = prompt
        st.session_state.assistant_messages = []
        with st.spinner("Generating answer..."):
            try:
                answer = ask_with_autogen(prompt)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            else:
                st.session_state.assistant_messages = [answer]

# Handle the follow-up action for generating research subtopics from the latest response.
elif subtopics_clicked:
    if not st.session_state.assistant_messages:
        st.warning("Please ask a question first.")
    else:
        last_response = st.session_state.assistant_messages[-1]
        prompt = (
            "Based on the latest response, generate 3 to 5 useful research subtopics and briefly "
            "describe each one in one sentence.\n\n"
            f"Context:\n{last_response}"
        )
        st.session_state.last_prompt = prompt
        st.session_state.assistant_messages = []
        with st.spinner("Generating subtopics..."):
            try:
                answer = ask_with_autogen(prompt)
            except Exception as exc:
                st.error(f"Subtopic generation failed: {exc}")
            else:
                st.session_state.assistant_messages = [answer]

# Handle the follow-up action for summarizing the latest response.
elif summarise_clicked:
    if not st.session_state.assistant_messages:
        st.warning("Please ask a question first.")
    else:
        last_response = st.session_state.assistant_messages[-1]
        prompt = (
            "Summarize the following research response in concise bullet points.\n\n"
            f"Response:\n{last_response}"
        )
        st.session_state.last_prompt = prompt
        st.session_state.assistant_messages = []
        with st.spinner("Summarizing response..."):
            try:
                answer = ask_with_autogen(prompt)
            except Exception as exc:
                st.error(f"Summarization failed: {exc}")
            else:
                st.session_state.assistant_messages = [answer]

# Show the latest prompt used for context in the current session.
if st.session_state.last_prompt:
    st.info(f"Last prompt: {st.session_state.last_prompt}")

# Render the generated assistant response and expose export actions.
if st.session_state.assistant_messages:
    st.markdown("### Assistant Response")
    for msg in st.session_state.assistant_messages:
        st.markdown(msg)

    # Build exportable documents for the current assistant output.
    doc = Document()
    for msg in st.session_state.assistant_messages:
        doc.add_paragraph(msg)
    word_io = BytesIO()
    doc.save(word_io)
    word_io.seek(0)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Assistant Responses"])
    for msg in st.session_state.assistant_messages:
        writer.writerow([msg])
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    csv_io = BytesIO(csv_bytes)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for msg in st.session_state.assistant_messages:
        pdf.multi_cell(0, 10, txt=msg)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_io = BytesIO(pdf_bytes)

    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.download_button(
            label="Download Word",
            data=word_io,
            file_name="assistant_response.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    with export_col2:
        st.download_button(
            label="Download CSV",
            data=csv_io,
            file_name="assistant_response.csv",
            mime="text/csv",
        )
    with export_col3:
        st.download_button(
            label="Download PDF",
            data=pdf_io,
            file_name="assistant_response.pdf",
            mime="application/pdf",
        )
