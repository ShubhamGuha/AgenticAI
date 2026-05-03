# Joke Generator with LangChain and LangServe

**Documented with the help of GitHub Copilot**

**Author:** Shubham Guha

## Overview

This project demonstrates a complete end-to-end implementation of an AI-powered joke generation system using modern Python frameworks and language models. It showcases how to build, deploy, and interact with LLM-based applications using LangChain, LangServe, and FastAPI.

## Components

1. **server.py**: A FastAPI server that hosts a joke generator chain using LangChain's composable components and Azure OpenAI's GPT-4 model
2. **client.py**: A command-line client that connects to the server via RemoteRunnable and generates jokes about user-specified topics

## Requirements

```bash
pip install langchain langchain-google-genai fastapi uvicorn langserve sse_starlette python-dotenv openai
```

## Installation & Setup

1. Clone or download this project
2. Create a virtual environment:
   ```bash
   python -m venv v_env
   ```
3. Activate the virtual environment:
   - **Windows (PowerShell)**: `v_env\Scripts\Activate.ps1`
   - **Windows (CMD)**: `v_env\Scripts\activate.bat`
   - **macOS/Linux**: `source v_env/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Configure environment variables:
   - Create a `.env` file in the project root
   - Add your Azure OpenAI credentials:
     ```
     AZURE_OPENAI_ENDPOINT=your_endpoint_url
     AZURE_OPENAI_API_KEY=your_api_key
     ```

## Running the Demo

### Step 1: Start the Server
In your first terminal:
```bash
python server.py
```
You should see: "Starting the Joke Generator API server..."

### Step 2: Run the Client
In a separate terminal, generate jokes about different topics:
```bash
python client.py --topic programming
```

### Try Different Topics
```bash
python client.py --topic food
python client.py --topic sports
python client.py --topic space
python client.py --topic animals
```

## Access the API

### Interactive API Documentation
Visit the auto-generated Swagger UI to interact with the API:
- **URL**: http://localhost:8000/docs

### Visual Playground
Use LangServe's built-in playground:
- **URL**: http://localhost:8000/joke-generator/playground/

### Direct API Access
You can also make HTTP requests directly:
```bash
curl -X POST http://localhost:8000/joke-generator/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"topic": "programming"}}'
```

## How It Works

This project demonstrates key LangChain and LangServe concepts:

### 1. **Prompt Engineering**
- Dynamic prompt templates that incorporate user input
- System and user messages for effective instruction

### 2. **LLM Integration**
- Integration with Azure OpenAI's GPT-4 model
- Temperature tuning for consistent output quality

### 3. **Chain Composition**
- Connecting prompts, models, and parsers using LangChain's pipe operator
- Clean, declarative syntax for building processing pipelines

### 4. **API Deployment**
- FastAPI framework for building robust REST APIs
- LangServe for automatically exposing chains as API endpoints
- Automatic documentation generation and interactive testing

### 5. **Client Communication**
- RemoteRunnable for seamless client-server interaction
- Clean abstraction over HTTP communication

## Architecture Flow

```
User Input (Topic)
    ↓
Client (client.py)
    ↓
HTTP Request → Server (server.py)
    ↓
Prompt Template (interpolate topic)
    ↓
Azure OpenAI GPT-4 Model
    ↓
String Parser
    ↓
Generated Joke
    ↓
HTTP Response → Client
    ↓
Display to User
```

## Project Structure

```
JokeGenerator/
├── server.py           # FastAPI server with joke generation chain
├── client.py           # CLI client for requesting jokes
├── requirements.txt    # Python package dependencies
├── README.md          # Project documentation (this file)
├── .env               # Environment variables (create this file)
└── v_env/             # Virtual environment directory
```

## Troubleshooting

- **Connection Error**: Ensure `server.py` is running before executing the client
- **API Key Error**: Verify that your Azure OpenAI credentials are correctly set in the `.env` file
- **Port Already in Use**: Change the port in `server.py` if 8000 is occupied
- **Import Error**: Ensure all requirements are installed in the activated virtual environment

## Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangServe Documentation](https://github.com/langchain-ai/langserve)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

## Future Enhancements

- Add authentication to the API
- Implement caching for frequently requested topics
- Add rate limiting
- Support multiple LLM providers
- Add logging and monitoring
- Create a web UI frontend

## License

This project is provided as-is for educational purposes.