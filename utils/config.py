import os 
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_ROUTER_API")
TIVALY_API_KEY = os.getenv("TAVILY_API")