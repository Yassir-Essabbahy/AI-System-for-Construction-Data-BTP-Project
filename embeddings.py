from pinecone import Pinecone
from config import PINECONE_API_KEY, EMBEDDING_MODEL

_pc = Pinecone(api_key=PINECONE_API_KEY)
_BATCH_SIZE = 96  # Pinecone inference API limit

def _embed(texts, input_type):
    if not texts:
        return []
    results = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i:i + _BATCH_SIZE]
        res = _pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=batch,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        results += [item["values"] for item in res.data]
    return results

def embed_documents(texts):
    return _embed(texts, "passage")

def embed_query(text):
    return _embed([text], "query")[0]