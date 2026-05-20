# Agentic RAG Competitive Analysis Agent

A CLI-based competitive analysis agent that uses Cohere and LlamaIndex to index competitor data and answer natural language queries using a ReAct-style workflow.

## Overview

- Loads or builds a vector index from `competitor_dataset.csv`
- Uses Cohere embeddings and generation via the installed `llama_index` Cohere wrappers
- Supports reasoning and tool use with a ReAct agent
- Includes a query history feature for the last 5 queries in the current session

## Files

- `competitive_analysis_agent.py`: main script for agent initialization and CLI interaction
- `competitor_dataset.csv`: sample competitor dataset used for indexing
- `requirements.txt`: Python dependencies
- `.env`: local environment file for API configuration (ignored by git)
- `tools/`: helper scripts for debugging, PDF extraction, environment checks, and imports

## Setup

1. Create and activate your Python virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Create a `.env` file in the project root with:

```text
COHERE_API_KEY=your_cohere_api_key_here
```

## Usage

Run the agent from the project root:

```bash
python competitive_analysis_agent.py
```

Then enter queries such as:

- `What are the main strengths of AlphaTech Solutions?`
- `Compare the marketing strategies of Competitor X and Competitor Y`
- `What is the average profit margin across competitors?`

Type `history` to view the most recent queries.

## Author

Shubham Guha
