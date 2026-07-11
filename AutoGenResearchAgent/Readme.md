# 📚 Research AI Agent

Author: Shubham Guha

## 🔍 Overview

This project is an AutoGen-powered research assistant that runs in a Streamlit app. Users can enter a broad topic, ask the agent to respond, generate subtopics, and summarize the latest output. The app uses Google Gemini through environment variables and supports exporting the generated content to Word, CSV, and PDF files.

---

## 🛠️ Features

- 🧠 Gemini-powered responses for research questions
- 📑 Subtopic generation from the latest response
- ✍️ Concise summarization of the current output
- 📥 Export options for Word, CSV, and PDF
- 🔐 Secure configuration through a project-local .env file
- 💬 Clear session-based interaction flow in the Streamlit interface

---

## 🧠 Tech Stack

- **Google Gemini API** – language generation
- **AutoGen** – multi-agent orchestration
- **Streamlit** – interactive UI
- **python-dotenv** – environment variable loading
- **python-docx / fpdf** – export support
- **Python** – core application logic

---

## 📦 Project Structure

```text
├── research_ai_agent.py  # Main Streamlit application entrypoint
├── .env                   # API key and model configuration (preferred source)
├── requirements.txt      # Python dependencies
└── Readme.md              # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AutoGenResearchAgent
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv autogen_venv
./autogen_venv/Scripts/activate   # On macOS/Linux: source autogen_venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a .env file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
MODEL_NAME=gemini-2.5-flash
```

The app reads configuration only from this file. No model_config.json file is required.

### 5. Run the Application

```bash
streamlit run research_ai_agent.py
```

The app is launched from research_ai_agent.py.

---

## 🧪 Notes

- Configuration is loaded from .env only.
- The project uses AutoGen's AssistantAgent and UserProxyAgent to demonstrate a simple research workflow in the Streamlit UI.
- The current imports resolve correctly inside the project virtual environment, so the app should launch with the pinned requirements shown in requirements.txt.

