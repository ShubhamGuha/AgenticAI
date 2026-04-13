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

# Load environment variables
load_dotenv()

# Configure Google AI API 
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not AZURE_API_KEY:
    st.error("Please set your AZURE_API_KEY in a .env file")
    st.stop()

#genai.configure(api_key=AZURE_API_KEY)
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT
)

# Setup embedding model
embedding_model = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large")

# Create or load Chroma vector store
VECTOR_STORE_DIR = "chroma_store"
if os.path.exists(VECTOR_STORE_DIR):
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# Extract text from uploaded files
def extract_text_from_legal_doc(file):
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# Store document analysis in vector store
def store_doc_analysis(resume_text, analysis, doc_id):
    documents = split_text(analysis)
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
    vectorstore.persist()

# Main App
def main():
    st.set_page_config(page_title="Legal Document Review App", layout="wide")
    st.title("Legal Document Review Application with LCEL and Vector Store")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Query or Document-Specific Question")
        doc_query = st.text_area("Enter your Document-Specific Query: ", height=300)
    with col2:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])

    if st.button("Analyze") and uploaded_file and doc_query:
        with st.spinner("Processing..."):
            legal_doc_text = extract_text_from_legal_doc(uploaded_file)
            with st.expander("View Document Text"):
                st.text(legal_doc_text)

            llm = AzureChatOpenAI(
                azure_deployment="gpt-4.1-mini",
                api_version="2024-02-15-preview",
                temperature=0.2
            )

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

            chain = (
                RunnableMap({
                    "doc_query": lambda x: x["doc_query"],
                    "legal_doc_text": lambda x: x["legal_doc_text"]
                })
                | prompt_template
                | llm
                | StrOutputParser()
            )

            analysis = chain.invoke({
                "doc_query": doc_query,
                "legal_doc_text": legal_doc_text
            })

            st.header("AI Analysis")
            st.markdown(analysis)

            # Store analysis in vector DB
            store_doc_analysis(legal_doc_text, analysis, os.path.splitext(uploaded_file.name)[0])
            st.success("Analysis stored in vector database.")

            st.download_button("Download Analysis", analysis, file_name="resume_analysis.txt")

if __name__ == "__main__":
    main()