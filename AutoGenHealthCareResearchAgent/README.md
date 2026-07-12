# AutoGen Healthcare Research Agent

This project provides a Streamlit-based healthcare research assistant that uses AutoGen and Gemini-backed model calls to answer clinical and medical research questions, summarize findings, compare treatments, and maintain compact short- and long-term memory across interactions.

## Features
- Research-style Q&A for healthcare topics
- Evidence-oriented summarization
- Treatment comparison workflows
- Session memory for recent and longer-term context
- Export of responses as Word, CSV, and PDF files

## Setup
1. Create or activate the project virtual environment.
2. Install dependencies:
   ```bash
   .\autogen_hca_venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Add your Gemini API key to a project-local `.env` file:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   MODEL_NAME=gemini-2.5-flash
   ```
4. Run the app:
   ```bash
   .\autogen_hca_venv\Scripts\python.exe -m streamlit run .\healthcare_research_agent.py
   ```

## Author
Shubham Guha
