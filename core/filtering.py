
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from core.faiss_store import store_in_faiss
from core.search import search_web
from utils.config import OPEN_ROUTER_API



llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API,
    model="gpt-3.5-turbo",
    temperature=0.25,
    max_tokens=2048
).with_config({"verbose": True})


# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Only extract content that is directly relevant to the query.
Query: {query}
Search Results: {results}

Return a concise text that is fully relevant.
"""
)

# Create an LLMChain
filter_chain = LLMChain(llm=llm, prompt=prompt_template)

def filter_and_store(query: str, num_results=3):
    # Step 1: get search results
    results = search_web(query, num_results=num_results)

    # Step 2: run the chain
    relevant_text = filter_chain.run(query=query, results=results)

    # Step 3: store in FAISS
    store_in_faiss(relevant_text)
    return relevant_text

