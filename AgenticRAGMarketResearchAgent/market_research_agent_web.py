"""
Market Research Agent Web Application

This application provides an intelligent market research agent that uses real-time web search
and Cohere's LLM for adaptive market research. It features an Agentic RAG architecture
with dynamic reasoning, adaptive retrieval, and document grounding.

Author: Shubham Guha
"""

import gradio as gr
import os
import requests
from bs4 import BeautifulSoup
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# DuckDuckGo search function
def web_search_duckduckgo(query, max_results=5):
    """
    Perform a web search using DuckDuckGo's HTML search interface.

    Args:
        query (str): The search query for market research
        max_results (int): Maximum number of search results to fetch (default: 5)

    Returns:
        list: List of text content from the top search result pages
    """
    DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    resp = requests.post(DUCKDUCKGO_SEARCH_URL, data={"q": query}, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", class_="result__a", limit=max_results)
    
    web_docs = []
    for link in links:
        href = link["href"]
        page = requests.get(href)
        text = BeautifulSoup(page.text, "html.parser").get_text()
        web_docs.append(text)
    return web_docs

# Function to query Cohere for reasoning
def agentic_rag_with_cohere(query, web_docs):
    """
    Use Cohere's RAG model to generate a reasoned response based on the query and retrieved documents.

    Args:
        query (str): The original market research query
        web_docs (list): List of text documents retrieved from web search

    Returns:
        str: The generated response from Cohere's model
    """
    formatted_docs = [{"text": doc} for doc in web_docs]
    response = co.chat(
        model="command-r-plus-08-2024",
        message=query,
        documents=formatted_docs,
    )
    return response.text

# Gradio Interface function
def market_research_agent(query):
    """
    Main function that orchestrates the market research process.

    Args:
        query (str): The user's market research query

    Returns:
        str: The agent's response with market research insights
    """
    web_docs = web_search_duckduckgo(query)
    response = agentic_rag_with_cohere(query, web_docs)
    return response

# Create the Gradio Interface
iface = gr.Interface(
    fn=market_research_agent, 
    inputs=gr.Textbox(label="Enter Your Market Research Query"), 
    outputs=gr.Textbox(label="Agent Response"),
    title="Market Research Agent",
    description="This agent uses real-time web search and Cohere's LLM for adaptive market research."
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch()