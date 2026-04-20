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

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
INTENT_DETECTION_NODE = "Intent Detection"

# ==================== STATE MANAGEMENT ====================
class CSRAgentState(TypedDict):
    """
    TypedDict defining the state structure for the LangGraph workflow.
    
    Attributes:
        user_input (str): Raw user message
        intent (Optional[str]): Classified intent (profile, incident, request, feature enhancement, unknown)
        data (Optional[dict]): Response data to be displayed to the user
        user_profile (Optional[Dict[str, str]]): User demographics (user_id, thread_id, email, contact_number)
        short_term_memory (Optional[Dict[str, str]]): In-session context (previous intents, last queries)
        long_term_memory (Optional[Dict[str, str]]): Cross-session persistence (past queries, history)
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
async def collect_user_data(state: CSRAgentState) -> CSRAgentState:
    """
    Extracts and updates user profile information from user input.
    
    Process:
    1. Uses LLM to extract user demographics (user_id, thread_id, email, contact_number)
    2. Updates user_profile if new information is detected
    3. Generates follow-up questions if profile data is incomplete
    
    Args:
        state (CSRAgentState): Current state containing user input and profile data
        
    Returns:
        CSRAgentState: Updated state with new profile data and response message
    """
    user_input = state['user_input']
    user_profile = state.get('user_profile', {})
    short_term_memory = state.get('short_term_memory', {})

    # Prompt LLM to extract user profile information
    prompt = (
        f"Extract user profile information (user_id, thread_id, email, contact_number) from: {user_input}. "
        f"Current profile: {user_profile}. "
        f"If no new information is provided, ask a question to gather missing data (e.g., 'what is your user ID?' or 'what is your query?')."
        f"Keep tone empathetic and clear."
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    # Parse and update profile if profile data is detected
    if "user_id:" in message.lower() or "thread_id:" in message.lower() or "email:" in message.lower() or "contact_number:" in message.lower():
        for line in message.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                user_profile[key.lower()] = value
    else:
        # Store clarifying questions in short-term memory
        short_term_memory['last_question'] = message

    return {**state, "user_profile": user_profile, "data": {"response": message}, "short_term_memory": short_term_memory}

# ==================== INTENT DETECTION ====================
async def detect_intent(state: CSRAgentState) -> CSRAgentState:
    """
    Classifies user input intent using LLM and flags high-risk queries.
    
    Intent Categories:
    - 'profile': User sharing/updating personal information
    - 'request': User requesting specific information or action
    - 'incident': User reporting an issue or problem
    - 'feature enhancement': User suggesting improvements or new features
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
        f"Classify the user's intent into one of: 'profile', 'request', 'incident', 'feature enhancement', or 'unknown'.\n"
        f"User input: {user_input}\n"
        f"Previous intent: {short_term_memory.get('previous_intent', 'none')}\n"
        f"Long-term context: {long_term_memory.get('last_advice', 'none')}\n"
        f"Intent:"
    )
    response = await llm.ainvoke(prompt)
    content = response.content.strip().lower()

    # Extract intent using regex
    match = re.search(r"(profile|request|incident|feature enhancement)", content)
    intent = match.group(1) if match else "unknown"
    short_term_memory['previous_intent'] = intent

    # Detect high-risk keywords for HITL review
    hitl_keywords = ["review", "approve", "apply", "plan change"]
    hitl_flag = any(keyword in user_input.lower() for keyword in hitl_keywords)

    return {**state, "intent": intent, "short_term_memory": short_term_memory, "hitl_flag": hitl_flag}

# =================== REQUEST INFORMATION HANDLER ====================
async def request_information(state: CSRAgentState) -> CSRAgentState:
    """
    Handles user requests for specific information based on detected intent.
    
    Process:
    1. Uses intent to determine the type of information requested
    2. Retrieves relevant information from mock database or generates response using LLM
    3. Returns response to user
    
    Args:
        state (CSRAgentState): Current state with detected intent and user input
        
    Returns:
        CSRAgentState: Updated state with response data
    """
    user_input = state['user_input']
    short_term_memory = state.get('short_term_memory', {})
    user_profile = state.get('user_profile', {})

    # Prompt LLM to mock request response with context
    prompt = (
        f"Mock responding to a user request based on: {user_input}. "
        f"Consider user profile: {user_profile}. "
        f"Reply with a helpful response."
    )

    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    short_term_memory['last_request'] = user_input
    return {**state, "data": {"response": message}, "short_term_memory": short_term_memory}

# ==================== ISSUE RESOLUTION HANDLER ====================
async def issue_resolution(state: CSRAgentState) -> CSRAgentState:
    """
    Handles user-reported issues by providing troubleshooting steps.
    
    Process:
    1. Analyzes user input to identify the issue
    2. Provides step-by-step troubleshooting guidance
    3. Escalates to human review if issue is complex or unresolved
    
    Args:
        state (CSRAgentState): Current state with user input and profile
    CSRAgentState: Updated state with response data
    """
    user_input = state['user_input']
    user_profile = state.get('user_profile', {})
    short_term_memory = state.get('short_term_memory', {})

    # Prompt LLM to generate issue resolution steps
    prompt = (
        f"Identify the issue from: {user_input}. "
        f"Consider user profile: {user_profile}. "
        f"Provide clear troubleshooting steps. "
        f"If the issue seems complex or unresolved, suggest escalation to human support. "
        f"Use clear, empathetic language. "
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

    short_term_memory['last_issue'] = user_input
    return {**state, "data": {"response": message}, "short_term_memory": short_term_memory}
        
# ==================== PERSONALIZED ADVICE ====================
async def provide_advice(state: CSRAgentState) -> CSRAgentState:
    """
    Provides personalized advice based on user profile and history.
    
    Process:
    1. Analyzes user query with context from profile and past advice
    2. Generates tailored advice suitable for users
    3. Stores advice in long-term memory for future reference
    
    Args:
        state (CSRAgentState): Current state with user input, profile, and memory
        
    Returns:
        CSRAgentState: Updated state with advice and updated long-term memory
    """
    user_input = state['user_input']
    user_profile = state.get('user_profile', {})
    long_term_memory = state.get('long_term_memory', {})

    # Prompt LLM to generate personalized advice
    prompt = (
        f"Provide personalized advice based on: {user_input}. "
        f"User profile: {user_profile}. "
        f"Previous advice: {long_term_memory.get('last_advice', 'none')}. "
        f"Use clear, empathetic language suitable for users with limited financial literacy."
    )
    response = await llm.ainvoke(prompt)
    message = response.content.strip()

# ==================== HUMAN-IN-THE-LOOP (HITL) ====================
async def human_in_the_loop(state: CSRAgentState) -> CSRAgentState:
    """
    Handles high-risk queries by escalating to human review.
    
    Triggered when:
    - Query contains Approval keywords (review, approve, apply, plan change, etc.)
    - Requires expert review before proceeding
    
    Args:
        state (CSRAgentState): Current state with high-risk user input
        
    Returns:
        CSRAgentState: Updated state with HITL review message
    """
    user_input = state['user_input']
    # Generate HITL escalation message
    prompt = (
        f"The query '{user_input}' has been flagged as Approval required. "
        f"This query requires review by a Customer Representative. "
        f"Please wait for Customer Representative's input before proceeding."
    )
    message = prompt
    return {**state, "data": {"response": message}}

# ==================== FALLBACK HANDLER ====================
async def fallback(state: CSRAgentState) -> CSRAgentState:
    """
    Handles unrecognized intents gracefully.
    
    Returns user-friendly error message with usage suggestions.
    
    Args:
        state (CSRAgentState): Current state (unused)
        
    Returns:
        CSRAgentState: Updated state with fallback message
    """
    message = "🤔 Sorry, I didn't understand. Try asking queries related to your profile or any requests or you have any issue or feature enhancements."
    return {**state, "data": {"response": message}}


# ==================== LANGGRAPH WORKFLOW CONSTRUCTION ====================
def get_next_node(state: CSRAgentState) -> str:
    """
    Routing logic for conditional edges in the LangGraph workflow.
    
    Routes to different nodes based on detected intent and flags:
    - Approval queries -> Human in the Loop
    - Valid intents -> Corresponding handler nodes
    - Invalid/unknown -> Fallback
    
    Args:
        state (CSRAgentState): Current state with intent and hitl_flag
        
    Returns:
        str: Next node name in the workflow
    """
    if state.get("hitl_flag", False):
        return "human_in_the_loop"
    valid_intents = ["profile", "request", "incident", "feature enhancement","support", "product", "office", "location"]
    return state["intent"] if state["intent"] in valid_intents else "fallback"

# Initialize StateGraph with CSRAgentState as the state schema
builder = StateGraph(CSRAgentState)

# Add all nodes to the graph
builder.add_node(INTENT_DETECTION_NODE, detect_intent)
builder.add_node("Collect User Data", collect_user_data)
builder.add_node("Request Information", request_information)
builder.add_node("Issue Resolution", issue_resolution)
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
        "request": "Request Information",
        "incident": "Issue Resolution",
        "feature enhancement": "Provide Advice",
        "human_in_the_loop": "Human in the Loop",
        "fallback": "Fallback"
    }
)

# Compile the graph into an executable workflow
CSR_bot = builder.compile()

# ==================== STREAMLIT UI SETUP ====================
# Configure Streamlit page metadata
st.set_page_config(page_title="TechTrend Innovations - CSR Bot", page_icon="💬", layout="centered")
st.title("TechTrend Innovations - Customer Support Representative")
st.caption("Your trusted assistant for all customer support needs.")

# Initialize session state for message history and user data persistence
if "messages" not in st.session_state:
    st.session_state.messages = []
if "long_term_memory" not in st.session_state:
    st.session_state.long_term_memory = {}
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

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

# Process user input through CSR bot
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
            final_state = asyncio.run(CSR_bot.ainvoke(state))
            bot_reply = final_state['data']['response']
            
            # Update session state with new profile and memory data
            st.session_state.user_profile = final_state.get('user_profile', {})
            st.session_state.long_term_memory = final_state.get('long_term_memory', {})
            
            # Display assistant response
            st.markdown(bot_reply)
    
    # Add assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})