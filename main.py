from utils.config import OPEN_ROUTER_API
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from tools import faiss_tool, tivaly_tool


# --- LLM ---
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API,
    model="gpt-4o-mini",
    temperature=0.25,
    max_tokens=2048
).with_config({"verbose": True})

# --- Tools ---
tools = [faiss_tool, tivaly_tool]

# --- Agent ---
retrieval_agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# --- Interactive loop ---
if __name__ == "__main__":
    print("Type 'quit' to exit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ("quit", "exit"):
            print("Exiting…")
            break
        try:
            answer = retrieval_agent.run(query)
            print("\n--- Final Answer ---")
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
