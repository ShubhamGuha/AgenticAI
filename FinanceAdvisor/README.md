# FinAdvise - Personal Finance Assistant

**An intelligent, conversational AI-powered finance assistant leveraging LangGraph, LLaMA 3.3, and real-time stock data.**

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Overview

FinAdvise is a Streamlit-based personal finance management system designed to simplify financial planning through an empathetic, conversational AI interface. It combines sophisticated intent detection, user profiling, real-time stock data integration, and memory management to provide personalized financial guidance while maintaining high safety standards through Human-in-the-Loop (HITL) mechanisms for high-risk queries.
### Key Capabilities
- 💬 **Conversational AI**: Chat-based interface powered by LLaMA 3.3 via Groq API
- 📊 **Real-Time Stock Data**: Live stock prices via Alpha Vantage API
- 👤 **Smart User Profiling**: Contextual understanding of age, income, goals, and risk tolerance
- 💾 **Dual Memory System**: Short-term (session) and long-term (persistent) memory for continuity
- 🚨 **Risk Management**: Human-in-the-Loop for high-risk financial queries
- 💰 **Multi-Purpose**: Stock tracking, expense management, budgeting, and financial advice

---

## 🎯 Features

### 1. **User Profile Collection**
- Incrementally gathers user demographics and financial preferences
- Detects and extracts: age, income, financial goals, risk tolerance
- Provides intelligent follow-up questions when data is incomplete
- Context-aware responses tailored to user circumstances

### 2. **Intent Detection**
Classifies user queries into actionable categories:
- **Profile**: Update personal financial information
- **Stock**: Stock price lookups and investment insights
- **Expense**: Expense tracking and categorization
- **Budget**: Budget summaries and financial overviews
- **Advice**: Personalized financial recommendations
- **Unknown**: Graceful fallback handling

### 3. **Real-Time Stock Information**
- Extracts stock symbols intelligently using NLP
- Fetches daily OHLCV data from Alpha Vantage API
- Provides risk-tailored investment insights
- Robust error handling for API limitations and invalid symbols
- Includes rate-limit detection and user-friendly error messages

### 4. **Memory Management**

#### Short-Term Memory (Session-Based)
- Previous intent history for conversational continuity
- Recent queries and user interactions
- Contextual information for current session

#### Long-Term Memory (Cross-Session)
- Historical financial advice for consistency
- User preferences and past decisions
- Persistent context across multiple conversations

### 5. **Human-in-the-Loop (HITL) Safety**
High-risk query detection for keywords:
- `liquidate`, `retirement`, `all my savings`, `entire portfolio`
- Flags queries requiring expert review
- Simulates escalation to human financial advisor

### 6. **Empathetic Communication**
- Jargon-free, accessible language for all financial literacy levels
- Clear, actionable guidance
- Supportive and encouraging tone

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.30+ |
| **LLM** | LLaMA 3.3-70B (via Groq API) |
| **Workflow** | LangGraph 0.0.35+ |
| **Framework** | LangChain 0.1.14+ |
| **Data Fetching** | Alpha Vantage API (Stock Data) |
| **Async** | asyncio, Python 3.8+ |
| **Configuration** | python-dotenv |

---

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **Virtual Environment**: Recommended (venv or conda)
- **API Keys**:
  - [Groq API Key](https://console.groq.com) - for LLaMA 3.3 access
  - [Alpha Vantage API Key](https://www.alphavantage.co/api/) - for stock data

---

## 🚀 Installation & Setup

### 1. Clone Repository (if applicable)
```bash
git clone <repository-url>
cd finadvise
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv v_env

# Using conda (alternative)
conda create -n finadvise python=3.10
conda activate finadvise
```

### 3. Activate Virtual Environment

**Windows:**
```bash
v_env\Scripts\activate
```

**macOS/Linux:**
```bash
source v_env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=gsk_your_key_here

# Alpha Vantage API Key (get from https://www.alphavantage.co/api/)
ALPHA_VANTAGE_API_KEY=your_key_here
```

**⚠️ Security Note**: Never commit `.env` to version control. Use `.gitignore` (already configured).

### 6. Run the Application
```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501` in your default browser.

---

## 📚 Usage Guide

### Basic Interactions

#### Profile Setup
```
User: "I'm 28 years old and earn $60,000 a year"
Bot: [Updates profile and asks about financial goals]

User: "I want to save for a house and have moderate risk tolerance"
Bot: [Completes profile, ready for personalized advice]
```

#### Stock Queries
```
User: "What's the price of Tesla stock?"
Bot: [Fetches latest TSLA price and provides risk-tailored advice]

User: "Show me Apple stock prices"
Bot: [Retrieves AAPL data with investment insights based on your profile]
```

#### Expense Tracking
```
User: "Add $50 for groceries and $30 for gas"
Bot: [Confirms expense entry and categorizes spending]
```

#### Budget Requests
```
User: "Show my monthly budget"
Bot: [Generates personalized budget summary based on profile]
```

#### Financial Advice
```
User: "How should I plan for retirement?"
Bot: [Provides tailored advice considering age, income, and risk tolerance]
```

#### High-Risk Query (HITL Triggered)
```
User: "Should I liquidate my entire portfolio?"
Bot: [Flags as high-risk] "This query requires review by a financial advisor..."
```

---

## 📁 Project Structure

```
finadvise/
├── app.py                      # Main Streamlit application
├── README.md                   # This file (documentation)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (⚠️ git-ignored)
├── .gitignore                  # Git ignore rules
└── v_env/                      # Virtual environment directory
    └── (dependencies installed here)
```

---

## 🔄 Workflow Architecture

```
User Input
    ↓
[Intent Detection] ← Context (short-term & long-term memory)
    ↓
├─→ [HITL Check] → High-Risk? → [Human Review Escalation]
└─→ [Intent Router]
    ├─→ [Profile Collection]
    ├─→ [Stock Info Retrieval] → [Alpha Vantage API]
    ├─→ [Expense Tracker]
    ├─→ [Budget Summary]
    ├─→ [Provide Advice]
    └─→ [Fallback Handler]
    ↓
[Response Generation]
    ↓
[Memory Update] → Update long-term & short-term memory
    ↓
User Display
```

---

## 🔐 Security & Safety Features

1. **Environment Variables**: API keys stored in `.env` (git-ignored)
2. **HITL Mechanism**: High-risk queries flagged for expert review
3. **Input Validation**: Stock symbols validated with regex
4. **Error Handling**: Graceful handling of API errors and timeouts
5. **Logging**: Comprehensive logging for debugging and monitoring
6. **Rate Limit Protection**: Detects and handles API rate limiting

---

## 📊 API Integration

### Alpha Vantage
- **Endpoint**: TIME_SERIES_DAILY
- **Features**: Daily stock data (Open, High, Low, Close, Volume)
- **Rate Limit**: 5 calls/minute (free tier)
- **Timeout**: 10 seconds per request

### Groq API
- **Model**: LLaMA 3.3-70B-Versatile
- **Purpose**: Intent detection, advice generation, data extraction
- **Async**: Full asyncio support for non-blocking calls

---

## 🛠️ Customization Guide

### Adding New Intent
1. Update intent detection prompt in `detect_intent()`
2. Add new node to graph: `builder.add_node()`
3. Update conditional edges routing
4. Implement handler function

### Modifying Memory Strategy
- Adjust `short_term_memory` keys for session context
- Expand `long_term_memory` for cross-session data
- Configure persistence strategy (current: session-based)

### Changing LLM Model
Update in configuration:
```python
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"  # Change model here
)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `GROQ_API_KEY not found` | Ensure `.env` file exists in project root |
| `Alpha Vantage rate limit` | Wait 1 minute before retrying stock queries |
| `Invalid stock symbol` | Use standard ticker symbols (e.g., AAPL, MSFT) |
| `Connection timeout` | Check internet connection and API endpoint status |

---

## 📋 Dependencies

```
streamlit>=1.30.0           # Web UI framework
langchain>=0.1.14           # LLM framework
langchain-groq>=0.1.4       # Groq integration
langgraph>=0.0.35           # Graph-based workflows
python-dotenv>=1.0.0        # Environment management
requests>=2.31.0            # HTTP client
```

See [requirements.txt](requirements.txt) for complete list with pinned versions.

---

## 👨‍💼 Author & Contributions

**Author**: [Shubham Guha](https://github.com/shubhamguha)

**Educational Background**: Trained through [Edureka](https://www.edureka.co/) courses on:
- Agentic AI & LLM Applications
- LangGraph & Workflow Orchestration
- LLaMA Model Fine-tuning
- AI/ML Project Development

**Development Support**: [GitHub Copilot](https://copilot.github.com/)
- Code suggestions and optimization
- Documentation generation
- Architecture guidance
- Testing and debugging

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🔗 Resources & References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Groq Console](https://console.groq.com/)
- [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)

---

## 💡 Future Enhancements

- [ ] Database integration for persistent expense tracking
- [ ] Portfolio analysis and rebalancing recommendations
- [ ] Tax optimization suggestions
- [ ] Multi-currency support
- [ ] Advanced charting and visualizations
- [ ] Machine learning-based expense categorization
- [ ] Integration with banking APIs
- [ ] Mobile app version

---

## 📞 Support & Feedback

For issues, suggestions, or contributions:
1. Open an [issue](issues) on GitHub
2. Submit a [pull request](pulls) with improvements
3. Reach out via email or LinkedIn

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Active Development