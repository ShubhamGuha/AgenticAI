# LangGraph Multi-Agent Chatbot Project

Author: Shubham Guha

This project demonstrates building a multi-agent chatbot using LangGraph. It features two agents: a travel advisor and a hotel advisor, allowing users to get travel and hotel recommendations through a conversational interface.

## Overview

The chatbot leverages the LangGraph library to define a stateful graph where each node represents an agent. The agents communicate with each other and the user to provide relevant information and recommendations. The project uses Azure OpenAI for natural language processing and incorporates tools for agent handoff.

## Features

-   **Multi-Agent Architecture:** Employs two distinct agents (travel advisor and hotel advisor) designed for specific tasks.
-   **Agent Handoff:** Allows seamless transfer of conversation between agents using dedicated handoff tools.
-   **LangGraph Integration:** Utilizes LangGraph to manage the state and transitions between agents.
-   **Azure OpenAI Model:** Uses Azure OpenAI via `langchain_openai` for natural language processing.
-   **Human-Readable Responses:** Prioritizes clear and informative responses from agents before transitioning to another agent.
-   **PDF Note:** The notebook uses Azure OpenAI instead of the PDF's Anthropic setup. It still implements the multi-agent travel/hotel bot architecture described in the problem statement.
-   **Memory Checkpointing:** Uses `MemorySaver` for state management and resuming conversations.
-   **Stream Updates:** Displays updates from the graph in real-time, providing visibility into the conversation flow.

## Requirements

-   Python 3.7+
-   `openai`
-   `langchain`
-   `langchain-openai`
-   `langgraph`
-   `python-dotenv`

You'll also need an Azure OpenAI API key. You must set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` environment variables with your Azure deployment values.

## Installation

1.  Google Colab:

   This project has been developed and run on Google Colab.
   For better development, switch your Runtime to GPU.

2.  Install the required Python packages:

    ```bash
    pip install openai langchain langchain-openai langgraph python-dotenv
    ```

3.  Set the Azure OpenAI API key:

    You'll need to configure your Azure OpenAI endpoint and API key. The code expects the Azure values to be present in the environment variables `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`. Set these variables accordingly. A typical way is to set them via the terminal:

    ```bash
    export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
    export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    ```

    Alternatively, create a `.env` file in the project root with:

    ```env
    AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
    AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
    ```

    ```python
    import os
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
    ```

    If on VS Code then you can also create a `.env` file in the root directory of the project and store the values there.

    **Important:** Replace the placeholder values with your actual Azure OpenAI endpoint and API key.

## Usage

1.  Run the Jupyter Notebook `Building Multi AI Agents Chatbots With LangGraph.ipynb`.

2.  Follow the instructions within the notebook to execute the code and interact with the multi-agent chatbot. The notebook includes a sample conversation flow where the user asks for travel recommendations, then requests hotel recommendations, and finally inquires about activities near a chosen hotel.

## Code Structure

-   **`Building Multi AI Agents Chatbots With LangGraph.ipynb`:**  Jupyter Notebook containing the entire implementation of the multi-agent chatbot. It includes defining the agents, the LangGraph graph, the tool definitions, and the conversation flow.

## Key Components

-   **`MultiAgentState`:** Defines the state of the conversation, including the message history and the last active agent.
-   **`get_travel_recommendations`:** A function providing travel recommendations based on user queries.
-   **`get_hotel_recommendations`:** A function providing hotel recommendations based on user queries.
-   **`make_handoff_tool`:** Creates a handoff tool that allows agents to pass the conversation back and forth.
-   **`travel_advisor` / `hotel_advisor`:** LangChain agents orchestrated through LangGraph.
-   **`travel_advisor`:** A LangChain agent responsible for travel recommendations.
-   **`hotel_advisor`:** A LangChain agent responsible for hotel recommendations.
-   **`call_travel_advisor`:** A function to invoke the travel advisor agent.
-   **`call_hotel_advisor`:** A function to invoke the hotel advisor agent.
-   **`human_node`:** A node representing the interaction with the user.
-   **`builder`:**  The `StateGraph` object that defines the flow of the conversation between agents and the user.
-   **`graph`:** The compiled LangGraph graph.

## Notes and Improvements

-   The current implementation of `get_travel_recommendations` and `get_hotel_recommendations` provides basic keyword-based recommendations.  These functions could be greatly enhanced by integrating with external APIs (e.g., travel booking APIs, hotel APIs) or databases to provide more dynamic and personalized recommendations.
-   Error handling and input validation could be added to improve the robustness of the system.
-   The prompts for the agents could be further refined to improve their performance and behavior.
-   The use of a real-time database or message queue could improve the scalability and responsiveness of the chatbot.
-   Consider adding more sophisticated tooling beyond simple keyword parsing to understand user intent.