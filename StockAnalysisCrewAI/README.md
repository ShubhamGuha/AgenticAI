# Stock Analysis Agent with CrewAI

## Overview
This project implements a multi-agent stock analysis workflow using CrewAI for Apple Inc. (AAPL). The notebook defines two specialized agents:
- **Stock Analyst**: retrieves recent news with DuckDuckGo and analyzes price history with Yahoo Finance.
- **Report Generator**: synthesizes the findings into a concise investor-style report.

The implementation is aligned with the requirement statement for a modular, scalable multi-agent system that uses external data sources and a Gemini LLM.

## Author
- **Author:** Shubham Guha

## Features
- Fetches recent stock news and 1-month price history.
- Analyzes price trends and technical highlights.
- Produces a professional investor report.
- Uses environment-based configuration for the Gemini API key.

## Prerequisites
- Python 3.8+
- A Google Gemini API key
- Internet access for Yahoo Finance and DuckDuckGo requests

## Project Structure
```text
StockAnalysisCrewAI/
├── .env                      # Local environment variables (API keys)
├── .gitignore                # Files excluded from GitHub
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── stock_analysis_agent_crew_ai.ipynb  # Main notebook implementation
└── saa_venv/                 # Local virtual environment (ignored by Git)
```

## Setup Instructions
1. Create and activate a virtual environment.
   ```bash
   python -m venv saa_venv
   .\saa_venv\Scripts\Activate.ps1
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file in the project root.
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Open and run the notebook sequentially.
   ```bash
   jupyter notebook stock_analysis_agent_crew_ai.ipynb
   ```

## Usage
1. Open the notebook in Jupyter.
2. Run the cells in order to:
   - load the API key,
   - define the custom tools and agents,
   - create tasks,
   - run the CrewAI workflow.
3. Review the generated report in the notebook output.

## Notes
- The notebook is designed to prompt for the Gemini API key if it is not already present in the environment.
- The workflow is intentionally modular so additional stocks or data sources can be added later.

## Acknowledgments
- CrewAI for the multi-agent framework
- Yahoo Finance and DuckDuckGo for data sources
- Google Gemini for LLM support