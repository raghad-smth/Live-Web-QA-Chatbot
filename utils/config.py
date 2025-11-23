import os 
from dotenv import load_dotenv
import sys
import os

# Add the root folder to path to import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API")
TIVALY_API_KEY = os.getenv("TAVILY_API")