from langchain.tools import Tool
from core.faiss_store import retrieve_from_faiss, store_and_return_tivaly



faiss_tool = Tool(
    name="FAISS Retrieval",
    func=retrieve_from_faiss,  
    description="Use this tool first thing to check if the answer to a user query is already stored locally in FAISS. Input is a question string, output is the closest stored documents."
)


tivaly_tool = Tool(
    name="Tivaly Search",
    func=store_and_return_tivaly,  
    description="Use this tool if the answer is not found in local FAISS memory. Input is a question string, output is the web search result."
)
