"""Powered Competitive Analysis Agent with Agentic RAG.

This script loads or builds an indexed Competitive rivals tactics vector store, 
attaches a Cohere LLM, and exposes a command-line tool for natural language Analysis queries.
"""

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import cohere
from llama_index.core import Document, StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.workflow import Context

# Configure logging for the application and library components.
# INFO-level logs provide a good balance between detail and readability.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env, if available.
# This allows the user to keep API keys out of source control.
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    logger.error(
        "COHERE_API_KEY not found. Set COHERE_API_KEY in .env or environment variables before running."
    )
    raise SystemExit("Missing COHERE_API_KEY environment variable. Export it or add it to .env.")

# Paths and configuration for the dataset and persisted index.
INDEX_STORAGE_DIR = Path("index_storage")
COMPETITOR_DATA_FILE = Path("competitor_dataset.csv")

# Initialize Cohere LLM and embeddings through the llama_index wrappers.
llm = Cohere(
    api_key=COHERE_API_KEY,
    model="command-r-plus-08-2024",
    temperature=0.2,
    max_tokens=2048,
)
embed_model = CohereEmbedding(
    api_key=COHERE_API_KEY,
    model="embed-english-v2.0"
)

def compute_analytics(metric: str, column: str, filter_condition: str = None) -> float:
    """Compute statistical metrics on the competitive dataset CSV using optional pandas filtering.

    This tool supports total/sum and average/mean calculations on numeric dataset columns.
    It verifies that the provided column exists and coerces values to numeric before aggregation.
    """
    if not COMPETITOR_DATA_FILE.exists():
        raise FileNotFoundError(f"Competitor data file not found: {COMPETITOR_DATA_FILE}")

    df = pd.read_csv(COMPETITOR_DATA_FILE)
    if filter_condition:
        try:
            df = df.query(filter_condition)
        except Exception as e:
            raise ValueError(f"Invalid filter_condition: {e}")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    # Convert the requested column to numeric and ignore non-numeric values.
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        raise ValueError(f"No numeric data found in column '{column}' after coercion.")

    if metric in ("sum", "total"):
        return float(series.sum())
    if metric in ("average", "mean"):
        return float(series.mean())

    raise ValueError("Unsupported metric. Use 'sum' or 'average'.")


analytics_tool = FunctionTool.from_defaults(
    fn=compute_analytics,
    name="analytics_tool",
    description="Computes statistical metrics (Total revenue or average growth) on Competitors revenue and growth data with optional filters.",
)

def build_or_load_index() -> VectorStoreIndex:
    """Load an existing index from disk, or create a new one from the competitor dataset CSV.

    The function persists the index to disk so subsequent runs reuse the vector store.
    """
    if INDEX_STORAGE_DIR.exists():
        logger.info("Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE_DIR))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    logger.info("Index not found. Building a new index from competitor_dataset.csv...")
    if not COMPETITOR_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Required competitive data file missing: {COMPETITOR_DATA_FILE}."
        )

    competitor_df = pd.read_csv(COMPETITOR_DATA_FILE)
    documents = []
    for _, row in competitor_df.iterrows():
        text = (
            f"CompetitorName: {row['CompetitorName']}, ProductDescription: {row['ProductDescription']}, "
            f"MarketingStrategy: {row['MarketingStrategy']}, Quarter: {row['Quarter']}, "
            f"Revenue: {row['Revenue (USD M)']}, ProfitMargin: {row['ProfitMargin (%)']}, YoYGrowth: {row['YoYGrowth (%)']}"
        )
        documents.append(Document(text=text))

    # Split long documents into chunks for better vector retrieval.
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter],
    )

    index.storage_context.persist(persist_dir=str(INDEX_STORAGE_DIR))
    logger.info("Index build complete and persisted to disk.")
    return index

"""Agent for analyzing competitive data using a ReAct framework."""
index = build_or_load_index()
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

competitive_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="competitive_data_tool",
    description="Provide Insights from the indexed competitive data. Use this tool to answer questions about competitor strength, weakness, performance, market trends, and sales tactics based on the dataset."
)

# Create the ReAct agent with a non-streaming mode because the current Cohere wrapper
# is not compatible with async streaming in this llama_index version.
agent = ReActAgent(
    tools=[competitive_tool, analytics_tool],
    llm=llm,
    system_prompt=(
        "You are a data scientist at a consulting company that assists businesses in acquiring a competitive advantage by examining their rivals' tactics. After using tools, you MUST provide a final answer starting with 'Answer: '."
    ),
    streaming=False,
    verbose=True,
    max_iterations=10,
)

class CompetitiveAnalysisAgent:
    """A wrapper class that exposes the ReAct agent as a simple query interface."""

    def __init__(self, agent: ReActAgent):
        self.agent = agent

    async def reason_and_act(self, query: str) -> str:
        """Process a natural language query using the ReAct agent and return the response."""
        logger.info("Starting agent reasoning for query: %s", query)
        try:
            # Run the agent and await its result.
            ctx = Context(self.agent)
            handler = self.agent.run(user_msg=query, ctx=ctx)
            result_obj = await handler
            logger.debug("Agent returned raw result object: %s", repr(result_obj))
            if hasattr(result_obj, "result"):
                response = result_obj.result()
            else:
                response = result_obj

            logger.info("Agent completed reasoning and tool execution.")
            # Extract the response content robustly.
            if hasattr(response, 'response'):
                result = str(response.response)
            elif hasattr(response, 'text'):
                result = str(response.text)
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)

            logger.info("Final response extracted from agent: %s", result)
            return result
        except Exception:
            logger.exception("Error in reason_and_act method.")
            raise

async def main():
    """Run the command-line interface loop for the competitive analysis agent."""
    try:
        query_history = []
        competitive_analysis_agent = CompetitiveAnalysisAgent(agent=agent)
        print("Welcome to Competitive Analysis Agent: Your AI-Powered Competitive Report Analysis Tool!")
        print("Enter your query (for example, 'What is the Q3 result of DeltaAI Labs 2025?' )")
        print("Type 'history' to view the last five queries, or 'exit' to quit.")

        while True:
            user_query = input("\nYour query: ").strip()
            if user_query.lower() == "exit":
                print("Exiting Competitive Analysis Agent. Goodbye!")
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
                response = await competitive_analysis_agent.reason_and_act(user_query)
                print(f"Response: {response}")
                # Track successful responses for history commands.
                query_history.append((user_query, str(response)))
            except Exception as error:
                logger.exception("Error processing query.")
                print(f"Error processing query: {error}")
                # Store failed query attempts too, for debugging.
                query_history.append((user_query, f"Error: {error}"))
    except KeyboardInterrupt:
        print("\nExiting Competitive Analysis Agent. Goodbye!")
    except Exception as e:
        logger.exception("Unexpected error in main loop.")
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Execute the async lifecycle correctly
    asyncio.run(main())