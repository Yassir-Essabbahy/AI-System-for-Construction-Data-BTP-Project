import os
from dotenv import load_dotenv

load_dotenv()

def _req(name):
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

GROQ_API_KEY       = _req("GROQ_API_KEY")
PINECONE_API_KEY   = _req("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "btp-ai")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION    = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "multilingual-e5-large")
EMBEDDING_DIM      = 1024
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K              = int(os.getenv("TOP_K", "5"))
MIN_SCORE          = float(os.getenv("MIN_SCORE", "0.5"))