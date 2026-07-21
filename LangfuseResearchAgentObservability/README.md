# Langfuse Research Agent Observability Demo

This project demonstrates observability for a LangGraph-based multi-agent research workflow using Langfuse, Groq, and DuckDuckGo. The notebook runs a three-step pipeline that searches for information, analyzes the results, and summarizes the findings while sending trace data to Langfuse.

## What the notebook does

- Loads API credentials from environment variables.
- Initializes a Groq client for LLM inference.
- Initializes a Langfuse client and callback handler for tracing.
- Builds a LangGraph workflow with three agents:
  1. Search agent – gathers recent web information with DuckDuckGo.
  2. Analyzer agent – extracts key points and trends.
  3. Summarizer agent – produces a concise report.
- Wraps the workflow inside a Langfuse trace span and attaches session metadata for observability and scoring.

## Requirements

- Python 3.10+
- Install dependencies from requirements.txt

## Setup

1. Create and activate a virtual environment.
2. Install the project dependencies:
   - pip install -r requirements.txt
3. Set the required environment variables:
   - GROQ_API_KEY
   - LANGFUSE_SECRET_KEY
   - LANGFUSE_PUBLIC_KEY
   - LANGFUSE_BASE_URL (optional, defaults to https://api.langfuse.com)
4. Run the notebook cells in order.

## Running the demo

1. Start the notebook in Jupyter, VS Code, or Google Colab.
2. Run the cells sequentially.
3. Enter a research topic when prompted.
4. Open the Langfuse dashboard to inspect the generated trace, session, and evaluation scores.

## Notes

- Keep your API keys secure and avoid committing them to Git.
- The demo is designed to work with a Langfuse project and organization configured in your environment.
- If the notebook cannot reach the API services, verify that your environment variables and network access are correct.