# AI Resume Screening with LangSmith Observability

A Streamlit-based resume screening application that analyzes uploaded resumes against job requirements using Gemini and tracks the full execution flow with LangSmith observability.

## Overview

This project helps recruiters or hiring teams quickly compare a candidate's resume against a role's requirements and produce a structured assessment with a suitability score.

## Features

- Upload resumes in PDF, DOCX, or TXT format
- Extract and preprocess resume content automatically
- Compare resume content to job requirements using Gemini
- Generate an HR-style structured analysis report
- Display a suitability score with visual feedback
- Store analysis chunks in a Chroma vector database
- Track runs and traces in LangSmith

## Project Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `chroma_store/` - Persisted Chroma vector database files
- `.env` - Environment variables for API keys
- `README.md` - Project documentation

## Setup Instructions

1. Create and activate a virtual environment

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables

   Create a `.env` file in the project root with the following values:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=resume-screening-demo
   ```

4. Run the application

   ```bash
   streamlit run app.py
   ```

## Author

Shubham Guha

