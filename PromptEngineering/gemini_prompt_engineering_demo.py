import os
import re
from typing import Optional

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env so the app can use the Gemini key and model.
load_dotenv()


st.set_page_config(page_title="Prompt Engineering Studio", page_icon="🧠", layout="wide")


def get_setting(name: str, default: Optional[str] = None) -> str:
    """Read an environment variable safely, falling back to an optional default."""
    return os.getenv(name, default or "") or ""


def build_model():
    """Create and configure the Gemini model instance from the current environment."""
    api_key = st.session_state.get("google_api_key") or get_setting("GOOGLE_API_KEY")
    model_id = st.session_state.get("model_id") or get_setting("GOOGLE_MODEL_ID", "gemini-2.5-flash")

    if not api_key or not model_id:
        raise ValueError("Set GOOGLE_API_KEY and GOOGLE_MODEL_ID in your environment or .env file.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_id)


def call_model(prompt: str, system_prompt: str = "", temperature: float = 0.2, max_tokens: int = 700, response_format=None):
    """Send a prompt to Gemini and return the model response text."""
    model = build_model()
    full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    response = model.generate_content(
        full_prompt,
        generation_config=generation_config,
    )
    return response.text


def extract_numeric_answer(text: str) -> str:
    """Extract the last number found in a text response, useful for simple demos."""
    if not text:
        return "?"
    matches = re.findall(r"\b\d+\b", text)
    return matches[-1] if matches else "?"


def render_section(title: str, description: str, prompt: str, system_prompt: str = ""):
    """Render one demo section with a prompt, a run button, and the returned response."""
    st.markdown(f"### {title}")
    st.caption(description)
    st.code(prompt, language="text")
    if st.button(f"Run {title}", key=title.replace(" ", "_").lower()):
        with st.spinner("Calling Gemini..."):
            try:
                response = call_model(prompt, system_prompt=system_prompt, temperature=0.2)
            except Exception as exc:
                st.error(f"Gemini API call failed: {exc}")
                return
        st.success("Response received")
        st.write(response)


def main():
    """Main Streamlit app entry point with one tab per prompt engineering technique."""
    st.title("🧠 Prompt Engineering Studio")
    st.write("A Streamlit demo of prompt engineering techniques using Google Gemini.")

    with st.sidebar:
        st.header("Gemini Settings")
        st.caption("Configure your Gemini API key and model before running the demos.")
        st.text_input("Gemini API key", key="google_api_key", type="password", value=get_setting("GOOGLE_API_KEY"))
        st.text_input("Model ID", key="model_id", value=get_setting("GOOGLE_MODEL_ID", "gemini-2.5-flash"))
        st.slider("Default temperature", 0.0, 1.0, 0.2, key="temperature")

    tabs = st.tabs([
        "1. Zero-shot",
        "2. Few-shot",
        "3. CoT",
        "4. Zero-shot CoT",
        "5. Self-consistency",
        "6. Tree of Thoughts",
        "7. ReAct",
        "8. Role prompting",
        "9. Meta prompting",
        "10. Structured output",
        "11. Prompt chaining",
    ])

    # Each tab demonstrates a different prompting technique with a fresh example.
    with tabs[0]:
        prompt = """Classify the following support ticket as one of: billing, delivery, product, refund, or other. Reply with one word only.

Ticket: The package arrived two days late and the tracking page never updated after dispatch."""
        render_section(
            "Zero-shot",
            "Ask the model to solve a task directly without showing examples.",
            prompt,
            system_prompt="You are a precise support-ticket classifier.",
        )

    with tabs[1]:
        prompt = """Label each request with the right intent.

Example 1:
Input: 'Can you send me a replacement charger for my keyboard?'
Output: replacement

Example 2:
Input: 'I need a refund because the app keeps crashing.'
Output: refund

Example 3:
Input: 'My subscription renewed twice this month.'
Output: billing

Now label these:
1. 'I paid twice and the receipt is missing.'
2. 'The smart bulb arrived broken and I want a replacement.'
3. 'The app says my password is invalid even though I changed it.'"""
        render_section(
            "Few-shot",
            "Give the model a few examples so it learns the desired pattern.",
            prompt,
        )

    with tabs[2]:
        prompt = """A small café buys 18 cartons of milk at $3.50 each. They sell 12 cartons for $5.00 each and the rest expire. If they throw away 3 cartons, how much profit did they make?

Solve this step by step before giving the final answer."""
        render_section(
            "Chain-of-Thought",
            "Show the model how to work through a multi-step problem before answering.",
            prompt,
            system_prompt="You are a careful arithmetic tutor. Show the steps clearly and finish with a final answer.",
        )

    with tabs[3]:
        prompt = """A gym has 40 lockers. 12 are already occupied, 7 are under repair, and 3 are reserved for staff. How many lockers are still available for members? Let's think step by step."""
        render_section(
            "Zero-shot CoT",
            "Add a lightweight reasoning trigger such as 'Let's think step by step.'",
            prompt,
        )

    with tabs[4]:
        # Self-consistency: ask the model several times and compare the answers.
        st.markdown("### Self-consistency")
        st.caption("Sample several possible reasoning paths and compare the answers.")
        prompt = """A delivery van travels 54 miles in the morning and 36 miles in the afternoon. It uses 3 gallons of fuel for the full day. What is the average miles per gallon for the trip? Think carefully and give the final number only."""
        st.code(prompt, language="text")
        if st.button("Run self-consistency demo", key="self_consistency"):
            with st.spinner("Sampling multiple reasoning paths..."):
                try:
                    answers = []
                    for i in range(4):
                        response = call_model(prompt, temperature=0.9)
                        answers.append(response.strip())
                    counts = {}
                    for answer in answers:
                        counts[answer] = counts.get(answer, 0) + 1
                    st.success("Completed 4 reasoning paths")
                    st.write("### Outputs")
                    for index, answer in enumerate(answers, 1):
                        st.write(f"{index}. {answer}")
                    st.write("### Majority vote")
                    st.write(counts)
                except Exception as exc:
                    st.error(f"Gemini API call failed: {exc}")

    with tabs[5]:
        prompt = """Solve this puzzle using Tree of Thoughts.

You need to make 24 using the numbers 2, 3, 4, 6 exactly once and the operators +, -, ×, ÷.

Instructions:
1. Generate 3 possible first moves.
2. For each, say whether it looks promising.
3. Continue the best branch to a full solution.
4. End with the final expression that makes 24."""
        render_section(
            "Tree of Thoughts",
            "Explore multiple branches of reasoning and prune weak ones.",
            prompt,
            system_prompt="You are a deliberate puzzle solver. Show branching ideas and then finish with the best solution.",
        )

    with tabs[6]:
        prompt = """Answer this question using the ReAct format.

Question: Which city is likely to have the lower monthly food cost: Seattle or Phoenix?

Use the format:
Thought: ...
Action: search('city cost data')
Observation: ...
Thought: ...
Final Answer: ..."""
        render_section(
            "ReAct",
            "Combine reasoning and tool-like actions in a loop.",
            prompt,
            system_prompt="You are a helpful research assistant. Follow the Thought → Action → Observation loop and keep the steps brief.",
        )

    with tabs[7]:
        prompt = """Explain how a city can reduce traffic congestion near schools without increasing costs too much."""
        render_section(
            "Role prompting",
            "Assign the model an expert persona to steer its tone and depth.",
            prompt,
            system_prompt="You are an experienced urban planning consultant who writes practical, concise recommendations for city leaders.",
        )

    with tabs[8]:
        prompt = """I need a prompt for a chatbot that helps customers choose between three phone plans.

Please create:
1. A strong system prompt
2. A short output format
3. One example of a good user question
4. A note about handling ambiguous requests"""
        render_section(
            "Meta prompting",
            "Ask the model to design or improve the prompt itself.",
            prompt,
            system_prompt="You are an expert prompt engineer who writes production-ready prompts.",
        )

    with tabs[9]:
        prompt = """Read this support email and return valid JSON only.

Email: I bought the smart speaker on Thursday, but the voice assistant never connects to Wi-Fi. I am angry and want a refund if this is not fixed this week.

Return this schema:
{
  "issue_category": "setup" | "billing" | "refund" | "device",
  "urgency": 1,
  "customer_sentiment": "angry" | "neutral" | "happy",
  "requires_refund": true,
  "summary": "short summary"
}"""
        render_section(
            "Structured output",
            "Force the model to return structured JSON instead of free-form text.",
            prompt,
            system_prompt="You are a support analyst. Return only valid JSON and do not include markdown fences.",
        )

    with tabs[10]:
        # Prompt chaining: use the output of one step as the input to the next.
        st.markdown("### Prompt chaining")
        st.caption("Break one task into smaller steps, passing the output of each step to the next.")
        article = """A local bakery opened a new delivery service this week. The first 500 orders were handled with a 96% on-time rate, but customers complained about missing labels on gluten-free items. The owner plans to add a barcode scanner next month to reduce mistakes."""
        st.code(article, language="text")
        if st.button("Run prompt chain", key="prompt_chain"):
            with st.spinner("Running chained prompts..."):
                try:
                    step1 = call_model(
                        f"Extract the main facts from this article into 3 bullet points.\n\n{article}",
                        system_prompt="You are a precise information extractor.",
                    )
                    step2 = call_model(
                        f"Using the bullet points below, write a single-sentence executive summary.\n\n{step1}",
                        system_prompt="You are a concise business writer.",
                    )
                except Exception as exc:
                    st.error(f"Gemini API call failed: {exc}")
                    return
            st.write("#### Step 1: Extract")
            st.write(step1)
            st.write("#### Step 2: Summarize")
            st.write(step2)


if __name__ == "__main__":
    main()
