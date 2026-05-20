"""InsightPulse sales analysis agent.

This script loads or builds an indexed sales vector store, attaches a Gemini LLM,
and exposes a command-line tool for natural language sales queries.

Note: aiohttp proxy patches are applied via sitecustomize.py before this module loads.
"""

import asyncio
import logging
import os
from pathlib import Path

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from google import genai
from llama_index.core import Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Configure logging for the application and library components
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow nested asyncio event loops for interactive execution environments
nest_asyncio.apply()

# Load environment variables from .env, if available
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure proxy settings to avoid aiohttp issues
# Remove proxy environment variables to disable proxy detection
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

if not GOOGLE_API_KEY:
    logger.warning(
        "GOOGLE_API_KEY not found. Set GOOGLE_API_KEY in .env or environment variables before running."
    )

# Paths and configuration
INDEX_STORAGE_DIR = Path("index_storage")
SALES_DATA_FILE = Path("sales_data.csv")

# Configure HTTP options for Google GenAI to avoid proxy issues
http_options = genai.types.HttpOptions(
    client_args={'trust_env': False}
)

# Initialize Gemini model and embedding model from environment configuration
# NOTE: Gemini models require a valid Google Cloud API key.
llm = GoogleGenAI(
    model_name="models/gemini-flash-latest", 
    api_key=GOOGLE_API_KEY,
    http_options=http_options
)
embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001", 
    api_key=GOOGLE_API_KEY,
    http_options=http_options
)


def compute_analytics(metric: str, column: str, filter_condition: str = None) -> float:
    """Compute statistical metrics on the sales CSV using optional pandas filtering."""
    if not SALES_DATA_FILE.exists():
        raise FileNotFoundError(f"Sales data file not found: {SALES_DATA_FILE}")

    df = pd.read_csv(SALES_DATA_FILE)
    if filter_condition:
        df = df.query(filter_condition)

    if metric == "sum":
        return float(df[column].sum())
    if metric == "average":
        return float(df[column].mean())

    raise ValueError("Unsupported metric. Use 'sum' or 'average'.")


analytics_tool = FunctionTool.from_defaults(
    fn=compute_analytics,
    name="analytics_tool",
    description="Computes statistical metrics (sum or average) on sales data with optional filters.",
)


def build_or_load_index() -> VectorStoreIndex:
    """Load an existing index from disk, or create a new one from the sales CSV."""
    if INDEX_STORAGE_DIR.exists():
        logger.info("Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE_DIR))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    logger.info("Index not found. Building a new index from sales_data.csv...")
    if not SALES_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Required sales data file missing: {SALES_DATA_FILE}."
        )

    sales_df = pd.read_csv(SALES_DATA_FILE)
    documents = []
    for _, row in sales_df.iterrows():
        text = (
            f"OrderID: {row['OrderID']}, Date: {row['Date']}, Region: {row['Region']}, "
            f"Product: {row['Product']}, Category: {row['Category']}, Quantity: {row['Quantity']}, "
            f"UnitPrice: {row['UnitPrice']}, TotalSale: {row['TotalSale']}"
        )
        documents.append(Document(text=text))

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter],
    )

    index.storage_context.persist(persist_dir=str(INDEX_STORAGE_DIR))
    logger.info("Index build complete and persisted to disk.")
    return index


index = build_or_load_index()
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

sales_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="sales_data_tool",
    description="Provides insights from sales data, including sales totals, product performance, and regional trends.",
)

agent = ReActAgent(
    tools=[sales_tool, analytics_tool],
    llm=llm,
    system_prompt=(
        "You are a sales assistant. After using tools, you MUST provide a final answer starting with 'Answer: '."
    ),
    verbose=True,
    max_iterations=10,
)


async def analyze_sales(query: str) -> str:
    """Send a natural language query to the ReAct agent and return the formatted response."""
    try:
        handler = agent.run(user_msg=query)
        response = await handler
        
        # Extract the response content
        if hasattr(response, 'response'):
            result = str(response.response)
        elif hasattr(response, '__str__'):
            result = str(response)
        else:
            result = str(response)
            
        return result
    except Exception as e:
        raise


if __name__ == "__main__":
    try:
        query_history = []
        print("Welcome to InsightPulse: Your AI-Powered Sales Report Analysis Tool!")
        print("Enter your query (for example, 'What is the total sales for Laptops in South in 2024?')")
        print("Type 'history' to view the last five queries, or 'exit' to quit.")

        while True:
            user_query = input("\nYour query: ").strip()
            if user_query.lower() == "exit":
                print("Exiting InsightPulse. Goodbye!")
                break
            if user_query.lower() == "history":
                if query_history:
                    print("\nRecent Queries:")
                    for i, (q, r) in enumerate(query_history[-5:], 1):
                        print(f"{i}. Query: {q}\n   Response: {r[:100]}...")
                else:
                    print("No query history yet.")
                continue
            if not user_query:
                print("Please enter a valid query.")
                continue

            print(f"\nProcessing query: {user_query}")
            try:
                response = asyncio.run(analyze_sales(user_query))
                print(f"Response: {response}")
                # Add successful query to history
                query_history.append((user_query, str(response)))
            except Exception as error:
                logger.exception("Error processing query.")
                print(f"Error processing query: {error}")
                # Add failed query to history
                query_history.append((user_query, f"Error: {error}"))
    except KeyboardInterrupt:
        print("\nExiting InsightPulse. Goodbye!")
    except Exception as e:
        logger.exception("Unexpected error in main loop.")
        print(f"Unexpected error: {e}")
