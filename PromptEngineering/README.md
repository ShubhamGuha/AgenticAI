# Prompt Engineering Studio

This project contains a Streamlit app that demonstrates common prompt engineering techniques using Google Gemini.

## What this app does

The app provides interactive demos for the following prompt engineering techniques:

1. Zero-Shot Prompting
2. Few-Shot Prompting
3. Chain-of-Thought (CoT)
4. Zero-Shot Chain-of-Thought
5. Self-Consistency
6. Tree of Thoughts (ToT)
7. ReAct Prompting
8. Role / Persona Prompting
9. Meta Prompting
10. Structured Output (JSON)
11. Prompt Chaining

## Files in this project

- [gemini_prompt_engineering_demo.py](gemini_prompt_engineering_demo.py) — Streamlit app with all demo sections
- [requirements.txt](requirements.txt) — Python dependencies
- [.env](.env) — environment variables for Gemini configuration

## Setup

1. Activate the virtual environment:
   ```powershell
   .\prompt_eng_venv\Scripts\activate
   ```

2. Install the required packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Make sure your [.env](.env) file contains:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_MODEL_ID=gemini-2.5-flash
   ```

## Run the app

Start the Streamlit app with:

```powershell
streamlit run gemini_prompt_engineering_demo.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## Prompt engineering techniques explained

### 1. Zero-Shot Prompting
Definition: Giving the model a task directly without showing examples. The model relies on its existing knowledge and instructions to solve the task.

### 2. Few-Shot Prompting
Definition: Providing a small number of input-output examples so the model can learn the pattern, format, or style you want.

### 3. Chain-of-Thought (CoT)
Definition: Encouraging the model to reason step by step before giving the final answer, which often improves performance on math and multi-step problems.

### 4. Zero-Shot Chain-of-Thought
Definition: Asking the model to reason step by step without providing examples, often by adding a trigger such as "Let's think step by step."

### 5. Self-Consistency
Definition: Asking the model to generate multiple reasoning paths and then choosing the most common answer, improving robustness on difficult tasks.

### 6. Tree of Thoughts (ToT)
Definition: Exploring multiple branches of reasoning, evaluating them, pruning weak ones, and backtracking to find a better solution.

### 7. ReAct Prompting
Definition: Combining reasoning and action by having the model think, take an action or tool call, observe the result, and continue iteratively.

### 8. Role / Persona Prompting
Definition: Assigning the model a specific expert identity or role so it answers in a more targeted tone, depth, or domain-specific style.

### 9. Meta Prompting
Definition: Prompting the model to write, improve, or optimize prompts for another task, effectively using the model to design better instructions.

### 10. Structured Output (JSON)
Definition: Instructing the model to return data in a strict, parseable structure such as JSON so it can be used in software applications.

### 11. Prompt Chaining
Definition: Breaking a complex task into multiple smaller prompts, where the output of one step becomes the input to the next step.

## Notes

- Some techniques may require more tokens or multiple model calls.
- If the Gemini API returns a quota error, wait and try again or use a different model or billing plan.
