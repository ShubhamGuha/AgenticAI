"""Streamlit code review assistant built with LangGraph and LangSmith."""

from __future__ import annotations

import ast
import os
import platform
import sys
import time
from typing import Any, TypedDict

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field


MAX_CODE_LENGTH = 5000
DEFAULT_MODEL = "gemini-2.5-flash"


class ReviewIssue(BaseModel):
    """A single actionable issue found during review."""

    category: str = Field(description="security, performance, style, or logic")
    severity: str = Field(description="critical, high, medium, low, or info")
    line: int | None = Field(default=None, description="1-based line number when known")
    finding: str
    recommendation: str


class CodeReview(BaseModel):
    """Stable response contract returned by the review model."""

    summary: str
    rating: int = Field(ge=0, le=10)
    issues: list[ReviewIssue]
    recommendations: list[str]
    corrected_examples: list[str] = Field(default_factory=list)


class ReviewState(TypedDict, total=False):
    code: str
    context: str
    validation_error: str | None
    review: CodeReview | None
    error: str | None
    started_at: float
    latency_ms: float


def configure_environment() -> tuple[str | None, str]:
    """Load local configuration and enable LangSmith tracing."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "code-review-assistant")

    # LangChain reads these variables automatically when it creates traced runs.
    if langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ.setdefault("LANGCHAIN_API_KEY", langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    return google_api_key, project


def validate_code(state: ReviewState) -> ReviewState:
    """Reject invalid input before spending a model request."""
    code = state.get("code", "")
    if not code.strip():
        return {"validation_error": "Empty code snippet."}
    if len(code) > MAX_CODE_LENGTH:
        return {"validation_error": f"Code snippet must be {MAX_CODE_LENGTH} characters or fewer."}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "the provided code"
        return {"validation_error": f"Invalid Python syntax at {location}: {exc.msg}."}
    return {"validation_error": None}


def route_after_validation(state: ReviewState) -> str:
    return "finish_invalid" if state.get("validation_error") else "review_code"


def review_code(state: ReviewState) -> ReviewState:
    """Ask Gemini for a structured review after local validation succeeds."""
    # Keep model imports inside the valid-input path so local validation can run
    # even in environments that block optional native dependencies.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return {"error": "GOOGLE_API_KEY is not configured."}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous senior Python reviewer. Analyze the supplied code "
                "for security, performance, PEP 8 style, and logic defects. Use only "
                "evidence in the code. Include line numbers when possible. Return a "
                "balanced review: do not invent issues, and mention strengths through "
                "the summary or recommendations.",
            ),
            (
                "human",
                "Application context: {context}\n\nPython code:\n```python\n{code}\n```",
            ),
        ]
    )
    # Structured output keeps the UI and scenario metrics independent of prose formatting.
    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        google_api_key=google_api_key,
        temperature=0.2,
    ).with_structured_output(CodeReview)
    chain = prompt | model
    try:
        review = chain.invoke({"code": state["code"], "context": state.get("context", "Not provided")})
        return {"review": review, "latency_ms": (time.perf_counter() - state["started_at"]) * 1000}
    except Exception as exc:
        return {"error": f"Model request failed: {exc}"}


def finish_invalid(state: ReviewState) -> ReviewState:
    return {
        "validation_error": state.get("validation_error"),
        "latency_ms": (time.perf_counter() - state["started_at"]) * 1000,
    }


def build_review_graph():
    # Import LangGraph only when a valid snippet is ready for model processing.
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ReviewState)
    graph.add_node("validate_code", validate_code)
    graph.add_node("review_code", review_code)
    graph.add_node("finish_invalid", finish_invalid)
    graph.add_edge(START, "validate_code")
    graph.add_conditional_edges(
        "validate_code",
        route_after_validation,
        {"review_code": "review_code", "finish_invalid": "finish_invalid"},
    )
    graph.add_edge("review_code", END)
    graph.add_edge("finish_invalid", END)
    return graph.compile()


def run_review(code: str, context: str = "") -> ReviewState:
    """Execute one complete, observable review request."""
    started_at = time.perf_counter()
    validation = validate_code({"code": code, "context": context, "started_at": started_at})
    if validation.get("validation_error"):
        # Invalid requests end before LangGraph, Gemini, or LangSmith network work.
        return finish_invalid({**validation, "started_at": started_at})

    from langsmith import traceable

    # The wrapper creates one top-level trace for each user or scenario request.
    @traceable(name="code_review_request", run_type="chain")
    def invoke_graph() -> ReviewState:
        return build_review_graph().invoke({"code": code, "context": context, "started_at": started_at})

    return invoke_graph()


TEST_CASES = {
    "Security issue": (
        "def run(user_input):\n    return eval(user_input)",
        "A command-line utility that evaluates user supplied expressions.",
    ),
    "Performance issue": (
        "def duplicates(items):\n    result = []\n    for item in items:\n        if item not in result:\n            result.append(item)\n    return result",
        "A data processing function called on large collections.",
    ),
    "Well-written code": (
        "def parse_port(value: str, default: int = 8080) -> int:\n    try:\n        port = int(value)\n    except ValueError:\n        return default\n    return port if 1 <= port <= 65535 else default",
        "A configuration helper for a web service.",
    ),
    "Empty input": ("", "A Python utility."),
    "Malformed syntax": ("def broken(:\n    return True", "A small helper function."),
}


def display_review(result: ReviewState) -> None:
    if result.get("validation_error"):
        st.error(result["validation_error"])
        return
    if result.get("error"):
        st.error(result["error"])
        return
    review = result.get("review")
    if not review:
        st.error("The assistant returned no review.")
        return
    st.metric("Quality rating", f"{review.rating}/10")
    st.markdown(f"**Summary**\n\n{review.summary}")
    if review.issues:
        st.subheader("Issues identified")
        st.dataframe([issue.model_dump() for issue in review.issues], use_container_width=True)
    st.subheader("Recommendations")
    for recommendation in review.recommendations:
        st.markdown(f"- {recommendation}")
    if review.corrected_examples:
        st.subheader("Corrected examples")
        for example in review.corrected_examples:
            st.code(example, language="python")
    st.caption(f"Completed in {result.get('latency_ms', 0):.0f} ms")


def main() -> None:
    google_api_key, langsmith_project = configure_environment()
    st.set_page_config(page_title="Code Review Assistant", page_icon="CR", layout="wide")
    st.title("Python Code Review Assistant")
    st.caption("LangGraph pipeline with Gemini analysis and LangSmith observability")

    with st.sidebar:
        st.header("Observability")
        st.write(f"Project: `{langsmith_project}`")
        if os.getenv("LANGCHAIN_API_KEY"):
            st.success("LangSmith tracing enabled")
            st.markdown(f"[Open LangSmith project](https://smith.langchain.com/projects/{langsmith_project})")
        else:
            st.warning("LANGCHAIN_API_KEY is not configured; tracing is disabled.")
        st.caption(f"Python {platform.python_version()} | {sys.platform}")

    if not google_api_key:
        st.error("Please set GOOGLE_API_KEY in .env before running a review.")
        st.stop()

    tab_review, tab_tests = st.tabs(["Review code", "Run test scenarios"])
    with tab_review:
        code = st.text_area("Python code", height=300, max_chars=MAX_CODE_LENGTH, placeholder="Paste a Python snippet...")
        context = st.text_input("Optional application context", placeholder="For example: authentication helper")
        if st.button("Analyze code", type="primary"):
            with st.spinner("Running validation and review..."):
                display_review(run_review(code, context))

    with tab_tests:
        selected = st.multiselect("Scenarios", list(TEST_CASES), default=list(TEST_CASES))
        if st.button("Run selected scenarios"):
            results: list[dict[str, Any]] = []
            for name in selected:
                scenario_code, scenario_context = TEST_CASES[name]
                result = run_review(scenario_code, scenario_context)
                # Keep validation failures in the report instead of treating them
                # as model successes, while still recording latency for every case.
                results.append(
                    {
                        "scenario": name,
                        "status": "success" if result.get("review") else "validation/error",
                        "latency_ms": round(result.get("latency_ms", 0)),
                        "review_length": len(result["review"].model_dump_json()) if result.get("review") else 0,
                    }
                )
            if results:
                st.dataframe(results, use_container_width=True)
                successful = sum(row["status"] == "success" for row in results)
                st.write(f"Completed {len(results)} scenarios; {successful} produced model reviews.")


if __name__ == "__main__":
    main()