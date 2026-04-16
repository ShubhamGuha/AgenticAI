"""
FinAdvise - Personal Finance Assistant
A Streamlit-based AI-powered finance assistant using LangGraph, LLaMA 3.3, and Alpha Vantage API.

Author: Shubham Guha
Contributions: Edureka Training, GitHub Copilot Documentation
Last Updated: April 2026
"""

import streamlit as st
import asyncio
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Dict
import re
from dotenv import load_dotenv
import os
import requests
import logging

# ==================== CONFIGURATION ====================
# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
INTENT_DETECTION_NODE = "Intent Detection"

# ==================== STATE MANAGEMENT ====================
class FinanceState(TypedDict):
    """
    TypedDict defining the state structure for the LangGraph workflow.
    
    Attributes:
        user_input (str): Raw user message
        intent (Optional[str]): Classified intent (profile, stock, expense, budget, advice, unknown)
        data (Optional[dict]): Response data to be displayed to the user
        user_profile (Optional[Dict[str, str]]): User demographics (age, income, goals, risk tolerance)
        short_term_memory (Optional[Dict[str, str]]): In-session context (previous intents, last queries)
        long_term_memory (Optional[Dict[str, str]]): Cross-session persistence (past advice, history)
        hitl_flag (Optional[bool]): Flag indicating high-risk queries requiring human review
    """
    user_input: str
    intent: Optional[str]
    data: Optional[dict]
    user_profile: Optional[Dict[str, str]]
    short_term_memory: Optional[Dict[str, str]]
    long_term_memory: Optional[Dict[str, str]]
    hitl_flag: Optional[bool]

# ==================== INITIALIZE LLM ====================
# Initialize Groq LLM with LLaMA 3.3 model (70B parameters)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# ==================== USER PROFILE COLLECTION ====================
async def collect_user_data(state: FinanceState) -> FinanceState:
    """
    Extracts and updates user profile information from user input.
    
    Process:
    1. Uses LLM to extract user demographics (age, income, goals, risk tolerance)
    2. Updates user_profile if new information is detected
    3. Generates follow-up questions if profile data is incomplete
    
    Args:
        state (FinanceState): Current state containing user input and profile data
        
    Returns:
        FinanceState: Updated state with new profile data and response message
    """
    user_input = state['user_input']
    user_profile = state.get('user_profile', {})
    short_term_memory = state.get('short_term_memory', {})

    # Prompt LLM to extract user profile information
    prompt = (
        f"Extract user profile information (age, income, financial goals, risk tolerance) from: {user_input}. "
        f"Current profile: {user_profile}. "
        f"If no new information is provided, ask a question to gather missing data (e.g., 'How old are you?' or 'What are your financial goals?'). "
        f"Keep tone empathetic and clear."
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    # Parse and update profile if profile data is detected
    if "age:" in message.lower() or "income:" in message.lower() or "goal:" in message.lower() or "risk:" in message.lower():
        for line in message.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                user_profile[key.lower()] = value
    else:
        # Store clarifying questions in short-term memory
        short_term_memory['last_question'] = message

    return {**state, "user_profile": user_profile, "data": {"response": message}, "short_term_memory": short_term_memory}

# ==================== INTENT DETECTION ====================
async def detect_intent(state: FinanceState) -> FinanceState:
    """
    Classifies user input intent using LLM and flags high-risk queries.
    
    Intent Categories:
    - 'profile': User sharing/updating personal information
    - 'stock': Stock price or investment queries
    - 'expense': Expense tracking requests
    - 'budget': Budget summary requests
    - 'advice': Financial advice requests
    - 'unknown': Unrecognized intent
    
    High-Risk Query Detection:
    - Flags queries containing keywords like "liquidate", "retirement", "all my savings"
    - Sets hitl_flag=True for HITL (Human-in-the-Loop) review
    
    Args:
        state (FinanceState): Current state with user input
        
    Returns:
        FinanceState: Updated state with detected intent and hitl_flag
    """
    user_input = state['user_input']
    short_term_memory = state.get('short_term_memory', {})
    long_term_memory = state.get('long_term_memory', {})

    # Prompt LLM to classify intent with context from memory
    prompt = (
        f"Classify the user's intent into one of: 'profile', 'stock', 'expense', 'budget', 'advice', or 'unknown'.\n"
        f"User input: {user_input}\n"
        f"Previous intent: {short_term_memory.get('previous_intent', 'none')}\n"
        f"Long-term context: {long_term_memory.get('last_advice', 'none')}\n"
        f"Intent:"
    )
    response = await llm.ainvoke(prompt)
    content = response.content.strip().lower()

    # Extract intent using regex
    match = re.search(r"(profile|stock|expense|budget|advice)", content)
    intent = match.group(1) if match else "unknown"
    short_term_memory['previous_intent'] = intent

    # Detect high-risk keywords for HITL review
    high_risk_keywords = ["liquidate", "retirement", "all my savings", "entire portfolio"]
    hitl_flag = any(keyword in user_input.lower() for keyword in high_risk_keywords)

    return {**state, "intent": intent, "short_term_memory": short_term_memory, "hitl_flag": hitl_flag}

# ==================== STOCK INFORMATION RETRIEVAL ====================
async def get_stock_info(state: FinanceState) -> FinanceState:
    """
    Fetches real-time stock data from Alpha Vantage API and provides risk-tailored advice.
    
    Process:
    1. Extracts stock symbol using LLM
    2. Validates symbol with regex (1-5 uppercase letters)
    3. Queries Alpha Vantage TIME_SERIES_DAILY endpoint
    4. Provides risk-tailored investment advice based on user profile
    5. Handles API errors, rate limits, and invalid symbols gracefully
    
    Args:
        state (FinanceState): Current state with user input and user_profile
        
    Returns:
        FinanceState: Updated state with stock price and advice, or error message
    """
    user_input = state['user_input']
    short_term_memory = state.get('short_term_memory', {})
    user_profile = state.get('user_profile', {})

    # Extract stock symbol using LLM with strict instructions
    prompt = (
        f"Extract the stock symbol (e.g., 'AAPL' for Apple) from the request: {user_input}. "
        f"Return only the symbol (e.g., 'AAPL') or 'UNKNOWN' if unclear. Do not include extra text."
    )
    response = await llm.ainvoke(prompt)
    stock_symbol = response.content.strip().upper()

    # Validate stock symbol with regex (1-5 uppercase letters)
    if not re.match(r'^[A-Z]{1,5}$', stock_symbol) or stock_symbol == 'UNKNOWN':
        message = f"Sorry, I couldn't identify a valid stock symbol from '{user_input}'. Please specify the stock (e.g., 'AAPL' for Apple)."
        logger.warning(f"Invalid stock symbol extracted: {stock_symbol}")
    else:
        # Query Alpha Vantage API for daily time series data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        try:
            # Fetch stock data with 10-second timeout
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Alpha Vantage API response for {stock_symbol}: {data.keys()}")

            # Extract closing price from latest trading day
            if "Time Series (Daily)" in data:
                latest_date = list(data["Time Series (Daily)"].keys())[0]
                stock_data = data["Time Series (Daily)"][latest_date]
                close_price = stock_data["4. close"]
                message = f"The latest closing price for {stock_symbol} is ${close_price} (as of {latest_date})."

                # Generate risk-tailored investment advice based on user profile
                risk_tolerance = user_profile.get('risk tolerance', 'unknown')
                risk_prompt = (
                    f"Provide a brief note on investing in {stock_symbol} tailored to a user with {risk_tolerance} risk tolerance. "
                    f"Keep it clear and empathetic."
                )
                risk_response = await llm.ainvoke(risk_prompt)
                message += f"\n{risk_response.content.strip()}"
            # Handle Alpha Vantage error responses
            elif "Error Message" in data:
                message = f"Error from Alpha Vantage: {data['Error Message']}. Please check the stock symbol or try again later."
                logger.error(f"Alpha Vantage error for {stock_symbol}: {data['Error Message']}")
            # Handle API rate limiting
            elif "Note" in data and "rate limit" in data["Note"].lower():
                message = "Alpha Vantage API rate limit exceeded. Please try again in a minute."
                logger.warning(f"Rate limit exceeded for {stock_symbol}: {data['Note']}")
            # Handle cases where stock symbol is valid but has no data
            else:
                message = f"No data available for {stock_symbol}. Please check the symbol or try again later."
                logger.error(f"No time series data for {stock_symbol}: {data}")
        except requests.RequestException as e:
            # Handle network errors and timeouts
            message = f"Error fetching data for {stock_symbol}: {str(e)}. Please try again later."
            logger.error(f"Request error for {stock_symbol}: {str(e)}")

    short_term_memory['last_stock_requested'] = user_input
    return {**state, "data": {"response": message}, "short_term_memory": short_term_memory}

# ==================== EXPENSE TRACKING ====================
async def track_expenses(state: FinanceState) -> FinanceState:
    """
    Mocks expense tracking and provides personalized confirmations.
    
    Process:
    1. Uses LLM to parse expense details from user input
    2. Considers user profile for context-aware responses
    3. Returns confirmation message
    
    Args:
        state (FinanceState): Current state with user input and profile
        
    Returns:
        FinanceState: Updated state with expense confirmation message
    """
    user_input = state['user_input']
    short_term_memory = state.get('short_term_memory', {})
    user_profile = state.get('user_profile', {})

    # Prompt LLM to mock expense addition with context
    prompt = (
        f"Mock adding an expense based on: {user_input}. "
        f"Consider user profile: {user_profile}. "
        f"Reply with a confirmation message, e.g., 'Added expense of $50 for groceries.'"
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    short_term_memory['last_expense'] = user_input
    return {**state, "data": {"response": message}, "short_term_memory": short_term_memory}

# ==================== BUDGET SUMMARY ====================
async def budget_summary(state: FinanceState) -> FinanceState:
    """
    Generates a mocked budget summary tailored to the user's profile.
    
    Process:
    1. Uses LLM to create realistic budget categories and totals
    2. Considers user profile (income, goals, etc.)
    3. Uses empathetic, clear language for financial clarity
    
    Args:
        state (FinanceState): Current state with user profile
        
    Returns:
        FinanceState: Updated state with budget summary message
    """
    user_profile = state.get('user_profile', {})
    
    # Prompt LLM to generate personalized budget summary
    prompt = (
        f"Mock a simple budget summary with categories and totals, tailored to user profile: {user_profile}. "
        f"Use clear, empathetic language."
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()
    return {**state, "data": {"response": message}}

# ==================== PERSONALIZED FINANCIAL ADVICE ====================
async def provide_advice(state: FinanceState) -> FinanceState:
    """
    Provides personalized financial advice based on user profile and history.
    
    Process:
    1. Analyzes user query with context from profile and past advice
    2. Generates tailored advice suitable for users with limited financial literacy
    3. Stores advice in long-term memory for future reference
    
    Args:
        state (FinanceState): Current state with user input, profile, and memory
        
    Returns:
        FinanceState: Updated state with advice and updated long-term memory
    """
    user_input = state['user_input']
    user_profile = state.get('user_profile', {})
    long_term_memory = state.get('long_term_memory', {})

    # Prompt LLM to generate personalized advice
    prompt = (
        f"Provide personalized financial advice based on: {user_input}. "
        f"User profile: {user_profile}. "
        f"Previous advice: {long_term_memory.get('last_advice', 'none')}. "
        f"Use clear, empathetic language suitable for users with limited financial literacy."
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    # Store advice in long-term memory for future reference
    long_term_memory['last_advice'] = message
    return {**state, "data": {"response": message}, "long_term_memory": long_term_memory}

# ==================== HUMAN-IN-THE-LOOP (HITL) ====================
async def human_in_the_loop(state: FinanceState) -> FinanceState:
    """
    Handles high-risk queries by escalating to human review.
    
    Triggered when:
    - Query contains high-risk keywords (liquidate, retirement, all savings, etc.)
    - Requires expert review before proceeding
    
    Args:
        state (FinanceState): Current state with high-risk user input
        
    Returns:
        FinanceState: Updated state with HITL review message
    """
    user_input = state['user_input']
    # Generate HITL escalation message
    prompt = (
        f"The query '{user_input}' has been flagged as high-risk. "
        f"This query requires review by a financial advisor. "
        f"Please wait for expert input before proceeding."
    )
    message = prompt
    return {**state, "data": {"response": message}}

# ==================== FALLBACK HANDLER ====================
async def fallback(state: FinanceState) -> FinanceState:
    """
    Handles unrecognized intents gracefully.
    
    Returns user-friendly error message with usage suggestions.
    
    Args:
        state (FinanceState): Current state (unused)
        
    Returns:
        FinanceState: Updated state with fallback message
    """
    message = "🤔 Sorry, I didn't understand. Try asking about stocks, expenses, budgets, or financial advice."
    return {**state, "data": {"response": message}}

# ==================== LANGGRAPH WORKFLOW CONSTRUCTION ====================
def get_next_node(state: FinanceState) -> str:
    """
    Routing logic for conditional edges in the LangGraph workflow.
    
    Routes to different nodes based on detected intent and flags:
    - High-risk queries -> Human in the Loop
    - Valid intents -> Corresponding handler nodes
    - Invalid/unknown -> Fallback
    
    Args:
        state (FinanceState): Current state with intent and hitl_flag
        
    Returns:
        str: Next node name in the workflow
    """
    if state.get("hitl_flag", False):
        return "human_in_the_loop"
    valid_intents = ["profile", "stock", "expense", "budget", "advice"]
    return state["intent"] if state["intent"] in valid_intents else "fallback"

# Initialize StateGraph with FinanceState as the state schema
builder = StateGraph(FinanceState)

# Add all nodes to the graph
builder.add_node(INTENT_DETECTION_NODE, detect_intent)
builder.add_node("Collect User Data", collect_user_data)
builder.add_node("Stock Info", get_stock_info)
builder.add_node("Expense Tracker", track_expenses)
builder.add_node("Budget Summary", budget_summary)
builder.add_node("Provide Advice", provide_advice)
builder.add_node("Human in the Loop", human_in_the_loop)
builder.add_node("Fallback", fallback)

# Set intent detection as the entry point
builder.set_entry_point(INTENT_DETECTION_NODE)

# Add conditional edges: Intent Detection -> Intent-specific nodes
builder.add_conditional_edges(
    INTENT_DETECTION_NODE,
    get_next_node,
    {
        "profile": "Collect User Data",
        "stock": "Stock Info",
        "expense": "Expense Tracker",
        "budget": "Budget Summary",
        "advice": "Provide Advice",
        "human_in_the_loop": "Human in the Loop",
        "fallback": "Fallback"
    }
)

# Compile the graph into an executable workflow
finance_bot = builder.compile()

# ==================== STREAMLIT UI SETUP ====================
# Configure Streamlit page metadata
st.set_page_config(page_title="💸 FinAdvise", page_icon="💬", layout="centered")
st.title("💸 FinAdvise")
st.caption("Your personal finance assistant for stocks, expenses, budgets, and tailored advice.")

# Initialize session state for message history and user data persistence
if "messages" not in st.session_state:
    st.session_state.messages = []
if "long_term_memory" not in st.session_state:
    st.session_state.long_term_memory = {}

# Display chat message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input and response loop
if user_input := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process user input through FinAdvise bot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Initialize state for workflow execution
            state = {
                "user_input": user_input,
                "intent": None,
                "data": None,
                "user_profile": st.session_state.get("user_profile", {}),
                "short_term_memory": {},
                "long_term_memory": st.session_state.long_term_memory,
                "hitl_flag": False
            }
            # Execute the LangGraph workflow asynchronously
            final_state = asyncio.run(finance_bot.ainvoke(state))
            bot_reply = final_state['data']['response']
            
            # Update session state with new profile and memory data
            st.session_state.user_profile = final_state.get('user_profile', {})
            st.session_state.long_term_memory = final_state.get('long_term_memory', {})
            
            # Display assistant response
            st.markdown(bot_reply)
    
    # Add assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})