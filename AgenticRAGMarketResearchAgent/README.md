# 🧠 Market Research Agent (Agentic RAG + Cohere + Web)

**Author:** Shubham Guha

This project is a **web-based intelligent agent** that performs **real-time market research** using:

* 🔍 **Live web search** (via DuckDuckGo)
* 🧠 **Cohere's Command-R+ LLM** for reasoning and summarization
* 📚 **Agentic RAG architecture**: dynamic reasoning, adaptive retrieval, and document grounding
* ✅ 100% open source and free-tier tools

---

## 📁 Folder Structure

```
AgenticRAGMarketResearchAgent/
├── market_research_agent_web.py       # Main agent logic with Gradio web interface
├── README.md                          # Project documentation
├── requirements.txt                   # Required dependencies
├── .gitignore                         # Git ignore file
├── .env                               # Contains API keys (not committed)
├── market_research_venv/              # Python virtual environment
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo or Navigate to Folder

Ensure you're in the project directory:

```
cd AgenticRAGMarketResearchAgent
```

### 2. Create `.env` file

Create a file named `.env` and add your Cohere API key:

```
COHERE_API_KEY=your_cohere_api_key_here
```

You can get a free API key from: [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)

### 3. Create and Activate Virtual Environment

Create a virtual environment:

```bash
python -m venv market_research_venv
```

Activate the virtual environment:

**Windows:**
```bash
market_research_venv\Scripts\activate
```

**macOS/Linux:**
```bash
source market_research_venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Agent

```bash
python market_research_agent_web.py
```

---

## 🧠 How It Works

1. **User enters a market research question** in the web interface
2. Agent performs a **live web search** using DuckDuckGo
3. Top 5 URLs are fetched, cleaned, and summarized
4. These are sent to **Cohere's RAG LLM (`command-r-plus`)** for reasoning
5. Agent returns a **concise, reasoned answer** in the web interface

### Example Interaction

```
🧑‍💼 Enter your market research question: What are the latest AI trends in retail?

📘 Retrieved and processed web content from top search results

🤖 Agentic RAG Answer:
"Recent AI trends in retail include the use of LLM-powered customer service chatbots, predictive inventory optimization, and hyper-personalized marketing through AI insights."
```

---

## 💡 Agentic RAG Concepts Used

* **Live retrieval** from external sources
* **Dynamic reasoning** based on real-time content
* **Multi-step agent architecture** (retrieval + generation)
* **Document-grounded generation** (not hallucinated)

---

## 🛠 Tech Stack

| Component      | Tool / API          |
| -------------- | ------------------- |
| Language Model | Cohere (Command-R+) |
| Web Search     | DuckDuckGo (HTML)   |
| Env Handling   | `python-dotenv`     |
| Parsing        | `beautifulsoup4`    |
| HTTP           | `requests`          |

---

## ✅ Requirements

```
cohere
requests
beautifulsoup4
python-dotenv
```

Installed via:

```bash
pip install -r requirements.txt
```

---

## 📌 To Do (Next Steps)

* [x] Gradio web interface
* [ ] Add conversation memory
* [ ] Add fallback search (Bing or Google via SerpAPI)
* [ ] PDF/CSV output generation
* [ ] Pluggable tools (charts, visual summaries)
* [ ] Error handling and logging improvements

---

## 📃 License

This project is open source and free to use for educational and non-commercial use.