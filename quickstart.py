from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant.

Rules:
- For every capital-city question, you MUST call `get_capital` before answering, even if you already know the answer.
- The tool returns structured data.
- If found=True, use capital from the tool and write Source: tool.
- If found=False and fallback_allowed=True, use your own knowledge and write Source: AI fallback.
- If found=False and fallback_allowed=False, do NOT use your own knowledge.
- In that case, write:
  Capital: Capital not found in tool
  Source: tool-not-found

Answer format:
Country: <country>
Capital: <capital or Capital not found in tool>
Source: <tool / AI fallback / tool-not-found>
"""

country_capitals = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Canada": "Ottawa",
    "Germany": "Berlin",
    "Australia": "Canberra",
    "Egypt": "Cairo",
    "Mexico": "Mexico City",
    "South Africa": "Pretoria",
}


@tool
def get_capital(country: str) -> dict:
    """Return structured capital information for a given country."""
    normalized = " ".join(word.capitalize() for word in country.strip().split())

    capital = country_capitals.get(normalized)

    if capital:
        return {
            "country": normalized,
            "found": True,
            "capital": capital,
            "fallback_allowed": False,
        }

    return {
        "country": normalized,
        "found": False,
        "capital": None,
        "fallback_allowed": normalized == "India",
    }


model = init_chat_model(
    "gemini-2.5-flash-lite",
    model_provider="google_genai",
    temperature=0.2,
    timeout=600,
    max_tokens=8000,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_capital],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = """
Answer each question with both the capital and the source.

1) What is the capital of India?
2) What is the capital of Brazil?
3) What is the capital of France?
4) What should you do if asked about India?
"""

agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "country-capital-simulation"}},
)

final_message = agent_result["messages"][-1]

print(final_message.content)