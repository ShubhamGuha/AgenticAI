"""
Customer Support Triage Agent

An AI-powered Streamlit application that automates customer support ticket triage using:
- Agno framework for AI agent orchestration
- Google Gemini LLM for intent/sentiment analysis and response generation
- Pinecone vector database for semantic search over support logs and policies
- Sentence Transformers for generating embeddings

Features:
- Multi-file upload (CSV, TXT, LOG, PDF)
- Automated ticket categorization and prioritization
- Intent and sentiment analysis
- Urgency detection
- Draft response generation
- Semantic search across support documents
- Dashboard with insights and chat history

Author: Shubham Guha
Date: May 22, 2026
"""

# Standard library imports
import io
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools import tool
from phi.workflow import Workflow
from pinecone import Pinecone, ServerlessSpec
import pinecone as _pinecone
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# Load API keys and configuration from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_ID = os.getenv("GOOGLE_MODEL_ID", "gemini-2.5-flash")  # LLM model ID
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone vector database configuration
INDEX_NAME = "support-triage"  # Name of the vector index
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"  # Sentence transformer model for embeddings
MAX_CHUNK_SIZE = 400  # Maximum characters per text chunk before splitting

# Initialize the Google Gemini LLM for analysis and response generation
model = Gemini(id=GOOGLE_MODEL_ID, api_key=GOOGLE_API_KEY)

# Initialize Pinecone vector database and create index if it doesn't exist
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
    existing_indexes = pc.list_indexes()
except Exception:
    existing_indexes = []

# Safely create the vector index; ignore conflicts if it was created concurrently
if INDEX_NAME not in existing_indexes:
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension matches the embedding model output
            metric="cosine",  # Use cosine similarity for semantic search
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except Exception as e:
        # If the index was created concurrently or already exists, ignore the conflict.
        err_str = str(e)
        if "ALREADY_EXISTS" in err_str or "409" in err_str:
            pass
        else:
            raise

# Get reference to the vector index for upserting and querying embeddings
index = pc.Index(INDEX_NAME)

# Load the sentence transformer embedding model used for semantic search and chunking
embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def safe_decode_bytes(data: bytes) -> str:
    """Safely decode bytes to string, ignoring encoding errors."""
    return data.decode("utf-8", errors="ignore")

def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV file from bytes, trying StringIO first, then BytesIO fallback."""
    try:
        return pd.read_csv(io.StringIO(safe_decode_bytes(file_bytes)))
    except Exception:
        return pd.read_csv(io.BytesIO(file_bytes))

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract all text from a PDF file provided as bytes."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or "\n"
    return text

def guess_columns(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    Intelligently guess which column matches the candidates.
    Tries exact matches first, then partial matches.
    """
    candidates = [c.lower() for c in candidates]
    for col in columns:
        if col.lower() in candidates:
            return col
    for col in columns:
        if any(candidate in col.lower() for candidate in candidates):
            return col
    return None

def parse_support_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract structured ticket records from a DataFrame.
    Attempts to identify issue, timestamp, and response columns automatically.
    Returns a list of dictionaries with standardized ticket fields.
    """
    issue_col = guess_columns(df.columns.tolist(), ["issue", "description", "problem", "ticket", "subject", "message", "complaint"])
    timestamp_col = guess_columns(df.columns.tolist(), ["timestamp", "date", "time", "created_at", "submitted"])
    response_col = guess_columns(df.columns.tolist(), ["response", "reply", "resolution", "agent_response", "answer"])

    if issue_col is None:
        issue_col = df.columns[0] if len(df.columns) > 0 else None
    records = []
    for _, row in df.iterrows():
        records.append({
            "issue": str(row[issue_col]) if issue_col is not None else "",
            "timestamp": str(row[timestamp_col]) if timestamp_col is not None else "",
            "response": str(row[response_col]) if response_col is not None else "",
            "source": row.to_dict(),
        })
    return records

def flatten_dataframe(df: pd.DataFrame) -> str:
    """
    Flatten a DataFrame into a single string representation.
    Joins rows with newlines and columns with pipes (|).
    Handles NaN values by converting them to empty strings.
    """
    rows: List[str] = []
    for _, row in df.iterrows():
        # Ensure every cell is string before joining to avoid type errors
        str_cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        rows.append(" | ".join(str_cells))
    return "\n".join(rows)

def chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = 80) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    Attempts to break at word boundaries to preserve semantics.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [chunk for chunk in chunks if chunk]

def index_document_chunks(source_name: str, text: str, doc_type: str) -> int:
    """
    Convert document text to embeddings and upsert to Pinecone in memory-safe batches.
    
    Handles large documents by:
    - Splitting into chunks with overlaps
    - Generating embeddings via sentence-transformers
    - Batching upserts to stay under Pinecone's 4MB message limit and 1000-vector limit
    - Trimming metadata when needed to fit constraints
    
    Args:
        source_name: Identifier for the document (filename)
        text: Document text to index
        doc_type: Type of document ('support_log', 'policy', 'support_text')
    
    Returns:
        Total number of vectors successfully upserted to Pinecone
    """
    chunks = chunk_text(text)
    vectors = []
    for index_i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vectors.append({
            "id": f"{source_name}_{doc_type}_{index_i}",
            "values": embedding,
            "metadata": {
                "source": source_name,
                "type": doc_type,
                "chunk_index": index_i,
                "snippet": chunk[:250],
            },
        })
    if not vectors:
        return 0

    # Pinecone / gRPC has a ~4MB message size limit for upserts. Estimate
    # per-vector bytes and build batches that stay safely under the limit.
    MAX_REQUEST_BYTES = 3_900_000  # conservative margin under 4MB

    def estimate_vector_bytes(v: dict) -> int:
        # Estimate embedding bytes (float64 ~8 bytes) and metadata/id size
        emb_len = len(v.get("values", []))
        emb_bytes = emb_len * 8
        meta = v.get("metadata", {})
        meta_bytes = 0
        for k, val in meta.items():
            meta_bytes += len(str(k)) + len(str(val))
        id_bytes = len(v.get("id", ""))
        # Small overhead per vector
        overhead = 200
        return emb_bytes + meta_bytes + id_bytes + overhead

    batch: List[dict] = []
    batch_bytes = 0
    sent = 0
    MAX_VECTORS_PER_REQUEST = 1000  # Pinecone hard limit per upsert

    for v in vectors:
        est = estimate_vector_bytes(v)
        # If single vector exceeds MAX_REQUEST_BYTES, fallback to very small batch sizes
        if est > MAX_REQUEST_BYTES:
            # try trimming snippet to reduce size
            v_copy = v.copy()
            if "metadata" in v_copy and "snippet" in v_copy["metadata"]:
                v_copy["metadata"]["snippet"] = v_copy["metadata"]["snippet"][:100]
                est = estimate_vector_bytes(v_copy)
            if est > MAX_REQUEST_BYTES:
                # last resort: send single vector (may still fail)
                try:
                    index.upsert(vectors=[v])
                    sent += 1
                    continue
                except Exception:
                    # raise if even single vector cannot be sent
                    raise

        if (batch_bytes + est > MAX_REQUEST_BYTES or len(batch) >= MAX_VECTORS_PER_REQUEST) and batch:
            index.upsert(vectors=batch)
            sent += len(batch)
            batch = []
            batch_bytes = 0

        batch.append(v)
        batch_bytes += est

    if batch:
        index.upsert(vectors=batch)
        sent += len(batch)

    return sent

# ============================================================================
# AGENT TOOLS
# ============================================================================
# These @tool-decorated functions expose LLM-callable operations for ticket triage
@tool
def sentiment_analysis(text: str) -> str:
    """Analyze the sentiment (positive, neutral, negative) of customer support text."""
    prompt = (
        "Read the following customer support text and return a concise sentiment label: positive, neutral, or negative. "
        f"If available, include a short rationale.\nText:\n{text}"
    )
    return model.generate(prompt).text

@tool
def intent_detection(text: str) -> str:
    """Classify the customer's intent (refund, delivery, account, product issue, etc.) from support text."""
    prompt = (
        "Classify the main customer intent from the following support text. "
        "Choose one or more categories like refund, delivery, account, product issue, shipping, billing, technical, or other. "
        f"Text:\n{text}"
    )
    return model.generate(prompt).text

@tool
def urgency_detection(text: str) -> str:
    """Determine the urgency level (high, medium, low) of a support ticket."""
    prompt = (
        "Analyze the following support text and return a single urgency level: high, medium, or low. "
        "Include any short justification if the text mentions deadlines, refunds, or escalations.\nText:\n"
        + text
    )
    return model.generate(prompt).text

@tool
def response_generation(text: str) -> str:
    """Generate a polite, concise draft response to a customer support issue."""
    prompt = (
        "Create a polite draft customer support response based on the following issue and context. "
        "Keep it concise and agent-ready.\nText:\n" + text
    )
    return model.generate(prompt).text

@tool
def summarize_common_issues(text: str) -> str:
    """Summarize frequent customer problems and trends from support data."""
    prompt = (
        "Summarize the most frequent customer support problems and trends from the text below. "
        f"Provide top themes and any recommended action items.\nText:\n{text}"
    )
    return model.generate(prompt).text

@tool
def search_similar(query: str) -> str:
    """Search for similar past support tickets using semantic similarity."""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    matches = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        snippet = metadata.get("snippet", "")
        source = metadata.get("source", "unknown")
        score = match.get("score", 0)
        matches.append(f"Source: {source}, Score: {score:.4f}\nSnippet: {snippet}")
    return "\n\n".join(matches)

@tool
def policy_reference(query: str) -> str:
    """Use indexed policies and past tickets to answer policy-related queries."""
    prompt = (
        "Use the most relevant policy or support text from the vector index to answer the query below. "
        f"Query:\n{query}"
    )
    search_results = search_similar(query)
    return model.generate(prompt + "\n\nRelevant snippets:\n" + search_results).text

class SupportTriageAgent:
    """
    Orchestrates triage workflow combining LLM analysis, semantic search, and tool invocation.
    
    Exposes the following capabilities:
    - Sentiment and urgency analysis
    - Intent classification
    - Draft response generation
    - Semantic search across indexed documents
    - Policy-based query answering
    """
    def __init__(self):
        self.agent = Agent(
            tools=[
                sentiment_analysis,
                intent_detection,
                urgency_detection,
                response_generation,
                summarize_common_issues,
                search_similar,
                policy_reference,
            ],
            model=model,
            name="Support Triage Agent",
            description=(
                "An AI agent that categorizes support tickets, extracts sentiment, urgency, and intent, "
                "generates draft responses, and performs semantic search over support logs and policy documents."
            ),
        )
        self.workflow = Workflow(agents=[self.agent], name="support_triage_workflow")

    def ask(self, query: str) -> str:
        """Run a natural language query against the triage agent and return results."""
        return self.agent.run(query)

# Initialize the agent instance for use in the Streamlit app
agent = SupportTriageAgent()

# ============================================================================
# STREAMLIT USER INTERFACE
# ============================================================================
# Configure page layout
st.set_page_config(page_title="Customer Support Triage Agent", layout="wide")
st.title("Customer Support Triage Agent")
st.write(
    "Upload support logs and policy documents, then ask the triage agent to categorize issues, find urgency, "
    "suggest replies, and surface related past tickets."
)

# Initialize session state variables for tracking uploads and conversation history
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # List of processed documents
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # Track filenames to avoid re-processing
if "summary_stats" not in st.session_state:
    st.session_state.summary_stats = {
        "total_files": 0,
        "total_chunks": 0,
        "documents": [],
    }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store Q&A pairs for display

# ============================================================================
# SIDEBAR: FILE UPLOAD AND QUERY INTERFACE
# ============================================================================
with st.sidebar:
    st.header("Upload support logs and policies")
    uploaded_files = st.file_uploader(
        "Choose one or more files:",
        type=["csv", "txt", "log", "pdf"],
        accept_multiple_files=True,
        help="Upload CSV support logs, TXT/log transcripts, or PDF policy documents.",
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip if already processed
            if uploaded_file.name in st.session_state.processed_files:
                continue
            try:
                file_bytes = uploaded_file.read()
                file_type = uploaded_file.name.split(".")[-1].lower()
                # Process file based on type
                if file_type == "csv":
                    df = load_csv_from_bytes(file_bytes)
                    text = flatten_dataframe(df)
                    records = parse_support_dataframe(df)
                    doc_type = "support_log"
                    summary_text = f"Processed CSV with {len(records)} ticket rows."
                elif file_type == "pdf":
                    text = extract_text_from_pdf_bytes(file_bytes)
                    records = []
                    doc_type = "policy"
                    summary_text = f"Processed PDF policy document with {len(text)} characters of extracted text."
                else:
                    text = safe_decode_bytes(file_bytes)
                    records = []
                    doc_type = "support_text"
                    summary_text = f"Processed text/log file with {len(text)} characters."

                chunk_count = index_document_chunks(uploaded_file.name, text, doc_type)
                st.session_state.uploaded_docs.append(
                    {
                        "name": uploaded_file.name,
                        "type": doc_type,
                        "size": uploaded_file.size,
                        "summary": summary_text,
                        "records": records,
                        "chunk_count": chunk_count,
                    }
                )
                st.session_state.processed_files.add(uploaded_file.name)
                st.session_state.summary_stats["total_files"] += 1
                st.session_state.summary_stats["total_chunks"] += chunk_count
                st.session_state.summary_stats["documents"].append(uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name} ({chunk_count} chunks)")
            except Exception as upload_error:
                st.error(f"Error indexing {uploaded_file.name}: {upload_error}")

    st.markdown("---")
    st.header("Ask the triage agent")
    # Natural language query interface
    query_prompt = st.text_input("Ask a question about uploaded support data", key="query_prompt")
    if st.button("Submit query"):
        if query_prompt.strip():
            with st.spinner("Analyzing support data..."):
                answer = agent.ask(query_prompt)
            st.session_state.chat_history.append({"question": query_prompt, "answer": answer})
            st.success("Query answered")

# ============================================================================
# MAIN DASHBOARD: METRICS, DOCUMENTS, CHAT, AND TOOLS
# ============================================================================

# Display key metrics at the top
col1, col2, col3 = st.columns(3)
col1.metric("Uploaded documents", len(st.session_state.uploaded_docs))
col2.metric("Indexed chunks", st.session_state.summary_stats["total_chunks"])
col3.metric("Workflow agents", 1)

# Section: Uploaded documents and their summaries
with st.expander("Uploaded documents and summaries", expanded=True):
    for doc in st.session_state.uploaded_docs:
        st.markdown(f"**{doc['name']}** — type: {doc['type']}, chunks: {doc['chunk_count']}")
        st.markdown(doc["summary"])
        if doc["records"]:
            st.write(f"Extracted {len(doc['records'])} structured rows from {doc['name']}")

# Section: Conversation history
with st.expander("Chat history", expanded=False):
    for entry in st.session_state.chat_history:
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}\n---")

# Section: Summary insights and statistics
with st.expander("Summary insights", expanded=True):
    if st.session_state.uploaded_docs:
        support_documents = [doc for doc in st.session_state.uploaded_docs if doc["type"] != "policy"]
        policy_documents = [doc for doc in st.session_state.uploaded_docs if doc["type"] == "policy"]
        st.write(f"**Support logs:** {len(support_documents)}")
        st.write(f"**Policy documents:** {len(policy_documents)}")
        st.write(f"**Indexed chunk count:** {st.session_state.summary_stats['total_chunks']}")
    else:
        st.info("Upload support logs or policy documents to enable semantic search and triage insights.")

# Section: Quick analysis tools
with st.expander("Quick tools", expanded=False):
    st.write("Use these quick tools on a sample ticket or policy text.")
    sample_text = st.text_area("Sample text for analysis", height=180)
    if st.button("Analyze sample text"):
        if not sample_text.strip():
            st.warning("Enter sample support text before analysis.")
        else:
            # Perform all analysis tools in parallel
            sentiment = sentiment_analysis(sample_text)
            intent = intent_detection(sample_text)
            urgency = urgency_detection(sample_text)
            response = response_generation(sample_text)
            st.write("**Sentiment:**")
            st.write(sentiment)
            st.write("**Intent:**")
            st.write(intent)
            st.write("**Urgency:**")
            st.write(urgency)
            st.write("**Draft response:**")
            st.write(response)

# Section: Semantic search interface
with st.expander("Semantic policy / issue search", expanded=False):
    st.write("Find similar past issues and relevant policy snippets using natural language.")
    semantic_query = st.text_input("Search across uploaded support content", key="semantic_query")
    if st.button("Search semantic index"):
        if semantic_query.strip():
            results = search_similar(semantic_query)
            st.text(results)
        else:
            st.warning("Enter a search query first.")
