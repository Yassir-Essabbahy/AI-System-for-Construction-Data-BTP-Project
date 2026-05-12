import hashlib
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIM, MIN_SCORE

_pc = Pinecone(api_key=PINECONE_API_KEY)

def ensure_index():
    if PINECONE_INDEX not in [i["name"] for i in _pc.list_indexes()]:
        _pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

def _index():
    return _pc.Index(PINECONE_INDEX)

def _make_id(metadata):
    key = (metadata.get("source", "") + str(metadata.get("page", "")) + metadata.get("text", ""))
    return hashlib.md5(key.encode()).hexdigest()

def upsert(records):
    if not records:
        return 0
    vectors = [{"id": _make_id(r["metadata"]), "values": r["values"], "metadata": r["metadata"]} for r in records]
    total = 0
    for i in range(0, len(vectors), 100):
        _index().upsert(vectors=vectors[i:i + 100])
        total += len(vectors[i:i + 100])
    return total

def query(vector, top_k):
    result = _index().query(vector=vector, top_k=top_k, include_metadata=True)
    matches = []
    for m in result.get("matches", []):
        if m.get("score", 0) < MIN_SCORE:
            continue
        meta = dict(m.get("metadata") or {})
        text = meta.pop("text", "")
        matches.append({"score": m.get("score", 0.0), "text": text, "metadata": meta})
    return matches

def clear():
    _index().delete(delete_all=True)