#!/usr/bin/env python
"""
Joke Generator API Server

This module implements a FastAPI server that provides a joke generation service using
LangChain for prompt management and chain orchestration, Azure OpenAI for LLM capabilities,
and LangServe for exposing the chain as REST API endpoints.

The server creates a pipeline that:
1. Accepts a topic from the user
2. Constructs a dynamic prompt with the topic
3. Sends it to Azure OpenAI's GPT-4 model
4. Returns the generated joke as a response

Author: Shubham Guha
Documented with help of GitHub Copilot
"""

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file into the application context
# This includes API keys and endpoints for secure credential management
load_dotenv()

# Retrieve Azure OpenAI credentials from environment variables
# These credentials are required for authenticating with Azure's OpenAI services
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize the Azure OpenAI client for making LLM API calls
# This client supports direct OpenAI API calls if needed for advanced operations
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT
)

# ============================================================================
# CHAIN COMPONENTS SETUP
# ============================================================================

# Step 1: Define the prompt template for joke generation
# The system message instructs the model on its role, while the user message
# requests a joke about a specific topic (dynamic parameter)
system_template = "You are a helpful assistant that generates jokes about {topic}."
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', 'Tell me a short, clean joke about {topic}.')
])

# Step 2: Initialize the Language Model
# Uses Azure's GPT-4 Mini model with low temperature (0.2) for more deterministic outputs
# Lower temperature = more consistent, less creative responses
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version="2024-02-15-preview",
    temperature=0.2
)

# Step 3: Initialize the output parser
# Converts the model's response into a clean string format
parser = StrOutputParser()

# Step 4: Create the LLM chain by piping components together
# The pipe operator (|) chains components: prompt -> llm -> parser
# This is LangChain's declarative syntax for building composition chains
chain = prompt_template | llm | parser

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="Joke Generator API",
    version="1.0",
    description="A simple demo API using LangChain and LangServe to generate jokes about a given topic",
)

# Register the chain as an API route using LangServe
# This automatically generates POST endpoints and provides interactive documentation
add_routes(
    app,
    chain,
    path="/joke-generator",
)

# Define the root endpoint to provide API information
@app.get("/")
async def root():
    """Root endpoint providing API information and available endpoints."""
    return {
        "message": "Welcome to the Joke Generator API",
        "endpoints": {
            "joke_generator": "/joke-generator",
            "docs": "/docs"
        }
    }

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting the Joke Generator API server...")
    print("Visit http://localhost:8000/docs to interact with the API")
    # Start the Uvicorn ASGI server on localhost port 8000
    uvicorn.run(app, host="localhost", port=8000)