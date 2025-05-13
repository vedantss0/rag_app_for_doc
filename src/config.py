import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_TYPE = os.getenv("LLM_TYPE", "gemini")  # gemini, openai, or other
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Vector store settings
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model settings
EMBEDDING_MODEL = "text-embedding-ada-002"  # Default OpenAI embedding model