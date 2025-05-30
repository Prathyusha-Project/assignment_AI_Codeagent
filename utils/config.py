import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Azure OpenAI Settings
    OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "azure")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")
    CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4")
    
    # Storage Paths
    CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./data/vector_db")
    REPOSITORIES_PATH = os.getenv("REPOSITORIES_PATH", "./data/repositories")
    
    # Processing Settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50000))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Pre-configured Repositories (Auto-load on startup)
    DEFAULT_REPOSITORIES = [
        "https://github.com/pallets/flask",
        # "https://github.com/psf/requests",
        # "https://github.com/tiangolo/fastapi",
    ]
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h', '.json', '.yaml', '.yml', 'docs', 'doc',}
    
    # Directories to exclude
    EXCLUDE_DIRS = {
        '.git', '__pycache__', 'node_modules', '.pytest_cache',
        'build', 'dist', '.venv', 'venv', 'env', '.mypy_cache',
        'tests', 'test',  'examples'
    }
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_KEY',
            'EMBEDDING_DEPLOYMENT_NAME'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Create directories if they don't exist
        Path(cls.CHROMADB_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
        Path(cls.REPOSITORIES_PATH).mkdir(parents=True, exist_ok=True)
        
        return True

# Validate configuration on import
Config.validate()