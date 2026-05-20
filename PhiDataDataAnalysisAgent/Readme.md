Data Analysis Agent with Agno
Author: Shubham Guha

Overview
--------
This Streamlit app uploads CSV, Excel, or PDF files and provides data insights using a combination of:
- Google Gemini for language understanding
- SentenceTransformers for embeddings
- Pinecone for semantic vector search
- Phidata/Agno for agent orchestration

Features
--------
- Upload CSV, Excel, or PDF files via Streamlit
- Generate pandas statistical summaries for CSV/Excel files
- Create text embeddings using `all-MiniLM-L6-v2`
- Store summary vectors in Pinecone for semantic search
- Retrieve similar dataset chunks using agent-based search

Requirements
------------
- Python 3.13
- `GOOGLE_API_KEY` in `.env`
- `PINECONE_API_KEY` in `.env`
- Install dependencies with `pip install -r requirements.txt`

Usage
-----
1. Activate the virtual environment
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `streamlit run data_analysis_agent_app.py`

Configuration
-------------
Add a `.env` file to the project root with:
```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
Optionally override the Gemini model with:
```
GOOGLE_MODEL_ID=gemini-2.5-flash
```

Project Files
-------------
- `data_analysis_agent_app.py` — main Streamlit application and agent logic
- `requirements.txt` — project dependencies
- `verify.py` — simple environment verification helper
- `.gitignore` — excluded files for Git
- `Readme.md` — project documentation

Notes
-----
- The app uses a supported Google Gemini model ID by default.
- `torchvision` should be installed to avoid CLI import errors from transformer extras.
