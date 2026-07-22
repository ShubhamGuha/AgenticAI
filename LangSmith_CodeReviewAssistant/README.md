# Python Code Review Assistant

**Author:** Shubham Guha

A Streamlit application that reviews Python code with Google Gemini, orchestrates
the workflow with LangGraph, and sends request traces to LangSmith.

## Features

- Validates that a snippet is non-empty, syntactically valid Python, and no more than 5,000 characters.
- Reviews security, performance, PEP 8 style, and logic concerns.
- Returns structured feedback with a rating, issue severity, line numbers, recommendations, and corrected examples.
- Accepts optional application context to make the review more relevant.
- Traces valid review requests with LangSmith, including model execution and latency.
- Includes five test scenarios for security, performance, good code, empty input, and malformed syntax.

## Setup

The project uses the included virtual environment. Activate it from the repository root:

```powershell
.\cra_venv\Scripts\activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root. The file must not be committed:

```env
GOOGLE_API_KEY=your-google-gemini-api-key
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=code-review-assistant
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
GEMINI_MODEL=gemini-2.5-flash
```

`GOOGLE_API_KEY` is required for model reviews. `LANGCHAIN_API_KEY` enables
LangSmith tracing; the application still starts without it, but tracing is disabled.

## Run

```powershell
streamlit run code_review_assistant.py
```

Streamlit opens the application in a browser. Use **Review code** for an individual
snippet or **Run test scenarios** to execute the built-in cases and view latency and
review-length metrics.

## Pipeline

The application uses a small LangGraph state machine:

1. `validate_code` checks the snippet locally with Python's `ast` parser.
2. Invalid input goes to `finish_invalid` and never calls Gemini.
3. Valid input goes to `review_code`, which invokes Gemini through a LangChain prompt chain.
4. Gemini returns the `CodeReview` Pydantic schema rather than unconstrained prose.
5. LangSmith traces the complete valid request and its model execution.

## Observability

Open the configured project in the LangSmith dashboard from the sidebar link. Traces
help inspect inputs, outputs, model failures, and latency. The test scenario table
also reports whether each case produced a model review and records its response size.

## Test Scenarios

| Scenario | Expected focus |
| --- | --- |
| Security issue | Detect unsafe use of `eval` and recommend safer parsing |
| Performance issue | Identify repeated list membership checks and complexity |
| Well-written code | Confirm strengths and avoid inventing defects |
| Empty input | Return local validation feedback without a model request |
| Malformed syntax | Report the syntax error location without a model request |

## Troubleshooting

If imports fail with an operating-system error mentioning the `xxhash` DLL, the
Python environment is blocking a native dependency used by LangSmith. This is an
environment policy issue rather than an application validation error. Allow the
virtual environment's native package binaries or use an approved Python environment,
then rerun the application.

## Files

- `code_review_assistant.py`: Streamlit application and LangGraph workflow.
- `requirements.txt`: Python dependencies.
- `.env`: Local API keys and configuration; ignored by Git.
- `project_doc.txt`: Assignment requirements.
- `sample.py`: Earlier LangSmith integration example.