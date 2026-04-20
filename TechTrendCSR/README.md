# TechTrend Innovations - CSR Agent

## Overview
An intelligent Customer Support Representative (CSR) chatbot built with **LangGraph**, **LangChain**, **Groq LLM**, and **Streamlit**. This agentic system handles customer inquiries through multi-turn conversations with intent detection, context awareness, and human-in-the-loop escalation.

## Features
- **Intent Detection**: Classifies user queries into 5 categories (profile, request, incident, feature enhancement, unknown)
- **Multi-Stage Workflow**: Streamlined processing with dedicated handlers for each intent type
- **Memory Management**: Dual-tier memory system for context retention (short-term & long-term)
- **User Profile Management**: Extracts and maintains user information across sessions
- **Human-in-the-Loop (HITL)**: Escalates high-risk queries to human review
- **Asynchronous Processing**: Non-blocking async/await for responsive user experience

## Architecture

### State Management (CSRAgentState)
```
├── user_input: Raw user message
├── intent: Classified intent category
├── data: Response payload
├── user_profile: Collected user demographics
├── short_term_memory: In-session context
├── long_term_memory: Cross-session persistence
└── hitl_flag: Human review trigger
```

### Workflow Nodes
1. **Intent Detection** (Entry Point): Analyzes user query and detects intent
2. **Collect User Data**: Extracts user profile information
3. **Request Information**: Handles information requests
4. **Issue Resolution**: Provides troubleshooting guidance
5. **Provide Advice**: Generates personalized recommendations
6. **Human in the Loop**: Escalates high-risk queries
7. **Fallback**: Handles unrecognized intents

## Installation

### Prerequisites
- Python 3.9+
- Groq API Key

### Setup
```bash
# Create virtual environment
python -m venv v_env
source v_env/bin/activate  # On Windows: v_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

## Usage

### Run the Application
```bash
streamlit run csr_app.py
```

The app will launch at `http://localhost:8501`

### Example Interactions
- **Profile Update**: "My email is john@example.com"
- **Information Request**: "What are your support hours?"
- **Issue Report**: "I can't log into my account"
- **Feature Request**: "Can you add dark mode?"

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key (required)

### Customization
- Modify `INTENT_DETECTION_NODE` in CONSTANTS section
- Add new intent handlers by creating functions and adding nodes
- Update `hitl_keywords` for custom escalation logic

## Project Structure
```
TechTrendCSR/
├── csr_app.py              # Main application file
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── README.md              # This file
└── v_env/                 # Virtual environment
```

## Key Dependencies
- **streamlit**: Web UI framework
- **langgraph**: Workflow orchestration
- **langchain-groq**: LLM integration
- **python-dotenv**: Environment management

## Author
**Shubham Guha**

## Version
1.0.0

## License
MIT License

## Support
For issues or feature requests, contact the development team.
