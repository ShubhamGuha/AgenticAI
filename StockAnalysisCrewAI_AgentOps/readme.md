# Stock Analysis with CrewAI and AgentOps

This project demonstrates a practical implementation of AgentOps observability for a CrewAI-based stock analysis workflow. The notebook integrates Gemini, DuckDuckGo, yFinance, and AgentOps to collect financial news, inspect recent price trends, and generate an investor-friendly report.

## Key Components
- CrewAI: Multi-agent orchestration for the analysis workflow.
- AgentOps: Monitoring and observability for agent runs, tools, and traces.
- Gemini LLM: Used as the reasoning model for the agents.
- Custom tools:
  - StockSearchTool: Searches recent stock-related news using DuckDuckGo.
  - YahooFinanceTool: Retrieves recent price history from yFinance.

## Workflow
1. Environment setup loads API keys from a local .env file.
2. Custom tools are defined for news retrieval and stock price history.
3. Two specialized agents are created: a Stock Analyst and a Report Generator.
4. Tasks are assigned for news gathering, trend analysis, and final report creation.
5. The agents and tasks are assembled into a CrewAI crew.
6. The crew is executed and the final report is printed.

## Setup
1. Clone the repository.
2. Install the required packages:
   `pip install -r requirements.txt`
3. Create a `.env` file with the following values:
   - `GEMINI_API_KEY=your_gemini_api_key_here`
   - `AGENTOPS_API_KEY=your_agentops_api_key_here`
4. Open and run the notebook in Jupyter or VS Code.

## Usage
Run the notebook:
`jupyter notebook AgentOps_Practical_Implementation.ipynb`

The workflow will generate a final stock analysis report for AAPL.

## Author
Shubham Guha
