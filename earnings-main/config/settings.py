import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Configuration
    GPT5_MODEL = "gpt-5"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # Token Limits (adjusted for GPT-5 reasoning)
    RESEARCH_AGENT_MAX_TOKENS = 2500
    VERIFICATION_AGENT_MAX_TOKENS = 1500

    # Retrieval Configuration
    CHUNK_SIZE = 1500  # Larger for financial tables
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 20  # Number of chunks to retrieve

    # Cache Configuration
    CACHE_DIR = Path.home() / ".cache" / "earnings_rag"
    CACHE_EXPIRY_DAYS = 7

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent

    def __init__(self):
        # Ensure cache directory exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Validate API keys
        if not self.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found in environment variables - set it before running the app")

settings = Settings()
