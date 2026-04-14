"""
Legal Document Review Application
Author: Shubham Guha (with assistance from Edureka training and GitHub Copilot)
Description: A Streamlit-based application that uses Azure OpenAI and LangChain 
             to analyze legal documents, extract key information, and provide 
             structured analysis based on user queries.
"""

import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from dotenv import load_dotenv
import shutil
import re

# ============================================================================
# INITIALIZATION AND CONFIGURATION SECTION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Configure Azure OpenAI API credentials
# These environment variables should be set in a .env file for security
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Validate that API key is configured; stop execution if not available
if not AZURE_API_KEY:
    st.error("Please set your AZURE_API_KEY in a .env file")
    st.stop()

# Initialize Azure OpenAI client with specified API version and deployment
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT
)

# ============================================================================
# VECTOR DATABASE INITIALIZATION
# ============================================================================

# Initialize embedding model for vectorizing text documents
# Using text-embedding-3-large for high-quality semantic representations
embedding_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large")

# Create or load persistent Chroma vector store for document embeddings
# The vector store is used to store and retrieve document analyses for future queries
VECTOR_STORE_DIR = "chroma_store"
if os.path.exists(VECTOR_STORE_DIR):
    # Load existing vector store if it already exists
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    # Create new vector store directory and initialize it
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_legal_doc(file):
    """
    Extract text content from uploaded legal documents.
    
    Supports multiple file formats: PDF, DOCX, and TXT files.
    The file is temporarily saved, processed, and then deleted to maintain memory efficiency.
    
    Args:
        file: Streamlit UploadedFile object containing the document to be processed
        
    Returns:
        str: Extracted text content from the document
        
    Raises:
        ValueError: If the file format is not supported (not PDF, DOCX, or TXT)
    """
    # Create a temporary file path to store the uploaded file
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    # Extract file extension to determine which loader to use
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        # Use appropriate document loader based on file type
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Load documents and extract text content
        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    finally:
        # Clean up temporary file regardless of success or failure
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def split_text(text):
    """
    Split text into smaller chunks for efficient processing and storage.
    
    Uses recursive character-based splitting to maintain semantic coherence between chunks.
    This is crucial for vector embeddings to capture meaningful context.
    
    Args:
        text (str): The complete text to be split into chunks
        
    Returns:
        list: List of Document objects with chunked text content
    """
    # Initialize text splitter with optimal chunk size and overlap
    # chunk_size=500: Each chunk contains approximately 500 characters
    # chunk_overlap=50: 50 characters overlap between consecutive chunks for context preservation
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])


def store_doc_analysis(resume_text, analysis, doc_id):
    """
    Store document analysis results in the Chroma vector store for future retrieval.
    
    This function vectorizes the analysis and stores it with unique document IDs,
    enabling semantic search capabilities for historical analyses.
    
    Args:
        resume_text (str): Original document text (parameter name is legacy, represents legal_doc_text)
        analysis (str): The AI-generated analysis of the document
        doc_id (str): Unique identifier for the document (typically the filename without extension)
        
    Returns:
        None: Stores data in vector database and persists to disk
    """
    # Split the analysis into manageable chunks for vector storage
    documents = split_text(analysis)
    
    # Add documents to vector store with unique IDs for tracking
    # Format: {doc_id}_chunk_{index} - this allows retrieval of specific chunks
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
    
    # Persist the vector store to disk for data durability
    vectorstore.persist()



# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main Streamlit application function.
    
    Orchestrates the entire user interface and application workflow:
    1. Configure page layout and title
    2. Create two-column layout for document upload and query input
    3. Process user input when "Analyze" button is clicked
    4. Generate AI-powered legal document analysis using LangChain
    5. Store results in vector database for future reference
    6. Provide analysis download capability
    """
    # Configure the Streamlit page with appropriate title and layout
    st.set_page_config(page_title="Legal Document Review App", layout="wide")
    st.title("Legal Document Review Application with LCEL and Vector Store")

    # Create two-column layout: left for query input, right for document upload
    col1, col2 = st.columns(2)
    
    # ========================================================================
    # LEFT COLUMN: Query Input
    # ========================================================================
    with col1:
        st.header("Query or Document-Specific Question")
        # Text area for users to enter specific questions about their legal documents
        doc_query = st.text_area("Enter your Document-Specific Query: ", height=300)
    
    # ========================================================================
    # RIGHT COLUMN: Document Upload
    # ========================================================================
    with col2:
        st.header("Upload Document")
        # File uploader supporting PDF, DOCX, and TXT formats
        uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])

    # ========================================================================
    # ANALYSIS PROCESSING
    # ========================================================================
    
    # Process analysis only when both query and file are provided
    if st.button("Analyze") and uploaded_file and doc_query:
        with st.spinner("Processing..."):
            # Extract text content from the uploaded document
            legal_doc_text = extract_text_from_legal_doc(uploaded_file)
            
            # Display the extracted document text in an expandable section
            with st.expander("View Document Text"):
                st.text(legal_doc_text)

            # Initialize Azure OpenAI LLM with optimized parameters for legal analysis
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4.1-mini",
                api_version="2024-02-15-preview",
                temperature=0.2  # Low temperature (0.2) for focused, consistent legal analysis
            )

            # Define the prompt template for legal document analysis
            # The template instructs the LLM to act as a legal expert and provide structured analysis
            prompt_template = PromptTemplate(
                input_variables=["doc_query", "legal_doc_text"],
                template="""
                You are an expert Legal Advisor and Professional. Understand and Analyze the legal document below against the query.

                Document-Specific Query:
                {doc_query}

                Resume:
                {legal_doc_text}

                Provide a structured analysis of how well the legal document understood, summarized, and query these documents. 
                At the end, clearly "summarize key information" based on Document-Specific Questions.
                Format: Key Clauses: <Bullet points>
                        Concise, Informative Summary: <Bullet points>
                """
            )

            # Build the LangChain runnable chain for document analysis
            # Uses LCEL (LangChain Expression Language) for composable pipelines
            chain = (
                RunnableMap({
                    # Map input variables to the prompt template
                    "doc_query": lambda x: x["doc_query"],
                    "legal_doc_text": lambda x: x["legal_doc_text"]
                })
                | prompt_template      # Pass mapped values to prompt
                | llm                   # Send prompt to LLM for analysis
                | StrOutputParser()     # Parse LLM output as string
            )

            # Execute the analysis chain with user inputs
            analysis = chain.invoke({
                "doc_query": doc_query,
                "legal_doc_text": legal_doc_text
            })

            # Display the AI-generated analysis results
            st.header("AI Analysis")
            st.markdown(analysis)

            # ================================================================
            # VECTOR STORE OPERATIONS
            # ================================================================
            
            # Store the analysis in the vector database for semantic search and future reference
            store_doc_analysis(legal_doc_text, analysis, os.path.splitext(uploaded_file.name)[0])
            st.success("Analysis stored in vector database.")

            # Provide download button for the analysis results
            st.download_button("Download Analysis", analysis, file_name="resume_analysis.txt")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Execute the main application
    main()