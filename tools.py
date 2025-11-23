from langchain.tools import Tool
from core.faiss_store import retrieve_from_faiss
from core.filtering import filter_and_store



faiss_tool = Tool(
    name="FAISS Retrieval",
    func=retrieve_from_faiss,  
    description="Use this tool first thing to check if the answer to a user query is already stored locally in FAISS. Input is a question string, output is the closest stored documents."
)


tivaly_tool = Tool(
    name="Tivaly Search",
    func=filter_and_store,  
    description="Use this tool if the answer is not found in local FAISS memory. Input is a question string, output is the web search result."
)
