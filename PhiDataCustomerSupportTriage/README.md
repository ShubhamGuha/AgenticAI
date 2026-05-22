# Customer Support Triage Agent

An AI-powered web application that automates customer support ticket triage using advanced NLP, semantic search, and agent orchestration.

## Overview

This application leverages the **Agno** (formerly PhiData) framework to build an intelligent support triage system that:

- **Automates ticket categorization** – Classifies support tickets by intent, sentiment, and urgency
- **Generates draft responses** – Suggests AI-powered customer responses
- **Performs semantic search** – Finds similar past issues across support logs
- **Analyzes trends** – Identifies common customer pain points and recurring issues
- **Integrates policies** – References company policies for consistent support

## Key Features

### 📤 Multi-File Upload
- CSV support logs and ticket databases
- TXT/LOG transcripts and chat histories
- PDF policy documents and compliance guides

### 🤖 AI-Powered Analysis
- **Sentiment Analysis**: Detect positive, neutral, or negative customer sentiment
- **Intent Detection**: Classify customer requests (refund, delivery, billing, technical, etc.)
- **Urgency Assessment**: Prioritize tickets by urgency level (high/medium/low)
- **Response Generation**: Create polite, concise draft responses

### 🔍 Semantic Search
- Search indexed documents using natural language
- Find similar past tickets and resolutions
- Retrieve relevant policy snippets using vector embeddings

### 📊 Dashboard & Insights
- Real-time metrics on uploaded documents and indexed chunks
- Chat history for query tracking
- Summary insights and document management

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | [Agno](https://github.com/phidatahq/phidata) (Agent Orchestration) |
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | Pinecone (Serverless, AWS) |
| **UI** | Streamlit |
| **Data Processing** | Pandas, pdfplumber |

## Installation & Setup

### Prerequisites
- Python 3.10+
- API Keys: Google (Gemini), Pinecone
- pip or conda package manager

### Step 1: Clone & Navigate
```bash
cd c:\Shubham\ProjectsAgenticAI\PhiDataCustomerSupportTriage
```

### Step 2: Create Virtual Environment
```bash
python -m venv cust_supp_venv
```

**Activate on Windows:**
```bash
cust_supp_venv\Scripts\activate
```

**Activate on macOS/Linux:**
```bash
source cust_supp_venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create a `.env` file in the project root with:
```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL_ID=gemini-2.5-flash
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Step 5: Run the Application
```bash
streamlit run customer_support_triage_app.py
```

The app will launch at `http://localhost:8501`

## Usage

### Upload Support Documents
1. Navigate to the **sidebar** → "Upload support logs and policies"
2. Select one or more files (CSV, TXT, LOG, PDF)
3. Files are automatically indexed and embedded into Pinecone

### Ask the Triage Agent
1. In the sidebar, enter a natural language query
2. Click **Submit query**
3. The agent analyzes indexed documents and returns insights

### Example Queries
- *"Summarize refund-related complaints from last week"*
- *"What are the top 3 customer pain points this month?"*
- *"Find all tickets with negative sentiment about shipping"*

### Quick Analysis
1. Go to the **Quick tools** section
2. Paste a sample support ticket
3. Click **Analyze sample text** to see:
   - Sentiment (positive/neutral/negative)
   - Intent (refund, delivery, account, etc.)
   - Urgency level
   - Draft response

### Semantic Search
1. Expand **Semantic policy / issue search**
2. Enter a search query (e.g., "password reset issues")
3. Results show similar indexed content with relevance scores

## Project Structure

```
PhiDataCustomerSupportTriage/
├── customer_support_triage_app.py    # Main Streamlit application
├── customer_support_tickets.csv      # Sample support ticket data
├── requirements.txt                   # Python dependencies
├── .env                               # API keys (not tracked by git)
├── .gitignore                         # Git ignore patterns
├── README.md                          # This file
└── cust_supp_venv/                   # Virtual environment
```

## Configuration

### Environment Variables
- `GOOGLE_API_KEY` – Google Cloud API key for Gemini LLM
- `GOOGLE_MODEL_ID` – Model ID (default: `gemini-2.5-flash`)
- `PINECONE_API_KEY` – Pinecone API key for vector storage

### Customizable Parameters (in code)
- `INDEX_NAME` – Pinecone index name (default: `support-triage`)
- `EMBEDDING_MODEL_ID` – Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `MAX_CHUNK_SIZE` – Text chunk size for indexing (default: 400 characters)

## Troubleshooting

### Error: "All support for google.generativeai has ended"
**Solution:** The package uses the newer `google-genai`. Ensure `requirements.txt` has the latest version:
```bash
pip install --upgrade google-genai
```

### Error: Pinecone "ALREADY_EXISTS" or "409"
**Solution:** The app safely handles concurrent index creation. If the error persists, manually delete the index in Pinecone console and restart the app.

### Error: "Payload Too Large" (413) or "decoded message length too large" (400)
**Solution:** The app uses adaptive batching to handle large documents. If issues persist:
- Reduce `MAX_CHUNK_SIZE` in the code
- Split large CSV files into smaller chunks
- Ensure PDF files aren't corrupted

### Error: File not found or encoding issues
**Solution:** The app attempts UTF-8 decoding with error tolerance. Try:
- Re-upload the file
- Ensure file is in one of the supported formats: CSV, TXT, LOG, PDF

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| pandas | Data manipulation |
| python-dotenv | Environment variable loading |
| pinecone-client | Vector database |
| sentence-transformers | Embedding generation |
| phidata | Agent framework (Agno) |
| google-genai | LLM API |
| pdfplumber | PDF text extraction |

Full list in `requirements.txt`

## Author

**Shubham Guha**  
Date: May 22, 2026

## License

This project is provided as-is for educational and internal use.

## Contributing

For improvements or bug reports, please document the issue and submit details.

## Future Enhancements

- [ ] Multi-language support
- [ ] Custom model fine-tuning
- [ ] Export analytics to CSV/PDF
- [ ] Webhook integrations for CRM/ticketing systems
- [ ] Real-time notification alerts
- [ ] Advanced filtering and sorting
- [ ] User authentication and role-based access

## Support

For issues, questions, or feature requests, contact the development team.
