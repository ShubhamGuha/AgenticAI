# InsightPulse Sales Analysis Agent

**Author:** Shubham Guha

InsightPulse is an AI-powered sales analysis tool that uses Gemini, a vector index, and an interactive command-line interface to answer natural language questions over sales data.

## Overview

This repository contains:
- `sales_analysis_agent.py` — Main application script for creating or loading the vector index and answering sales queries.
- `sales_data.csv` — Sales dataset used for analysis.
- `requirements.txt` — Python dependency list.
- `index_storage/` — Persisted index files used to speed up repeated runs.
- `.gitignore` — Ignored files and directories for Git.

## Author

Shubham Guha

## Documentation

This documentation was refined with the help of GitHub Copilot to make installation and usage clearer.

## Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Google Cloud API key with access to Gemini models

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv sales_analyzer_env
sales_analyzer_env\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API key:

```text
GOOGLE_API_KEY=<your-google-api-key>
```

4. Verify `sales_data.csv` is present in the project root. If you need to generate it, run the relevant data generator script.

## Running the Application

```powershell
python sales_analysis_agent.py
```

Then enter natural language questions like:
- `What is the total sales for Laptops in South in 2024?`
- `What is the average unit price of Monitors?`

Type `history` to view the last five queries. Type `exit` to quit.

## Notes

- The first run may take longer because it builds the vector index.
- The index is saved to `index_storage/` for faster future execution.
- If `sales_data.csv` is missing, the script will raise a clear error message.

## Troubleshooting

- **Missing API key:** Confirm `GOOGLE_API_KEY` is set in `.env` or your environment variables.
- **Missing data file:** Ensure `sales_data.csv` exists in the root folder.
- **Dependency issues:** Run `pip install -r requirements.txt` again.
- **Model warning:** If you see a warning about `google.generativeai`, the library is still supported for your current environment, but future upgrades should move to `google.genai`.

## Recommended `.gitignore`

A `.gitignore` file has been added to exclude environment files, caches, generated index data, and other artifacts from Git.

## License

This repository is intended for personal use and demonstration.
