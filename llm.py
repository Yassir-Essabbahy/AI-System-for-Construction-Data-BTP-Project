from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# the exact string the model is told to return when it has no answer
_NO_INFO_MARKER = "i don't have enough information in the provided documents"

SYSTEM_PROMPT = (
    "You are an AI assistant for a construction company (BTP). "
    "You analyze technical documents and professional emails. "
    "Rules:\n"
    "1. Use ONLY the context provided. Do not use outside knowledge.\n"
    "2. If the answer is not in the context, reply EXACTLY: "
    "\"I don't have enough information in the provided documents.\"\n"
    "3. Be concise, professional, and cite the source number like [1], [2] when you rely on a snippet.\n"
    "4. Answer in the same language as the question.\n"
    "5. If the question is not related to construction or the provided documents, reply EXACTLY: "
    "\"I don't have enough information in the provided documents.\"\n"
)

_SOURCE_KEYS = {
    "email": ["source", "type", "sender", "date", "project", "lot", "criticite"],
    "pdf":   ["source", "type", "page", "project", "lot", "criticite"],
    "text":  ["source", "type", "project", "lot", "criticite"],
}

def build_prompt(question, matches):
    if not matches:
        return f"Context: (none)\n\nQuestion: {question}"

    snippets = []
    for i, m in enumerate(matches, start=1):
        meta = m["metadata"]
        tags = ", ".join(f"{k}: {v}" for k, v in meta.items() if k != "type" and v not in (None, "", "Non defini"))
        header = f"[{i}]" + (f" ({tags})" if tags else "")
        snippets.append(f"{header}\n{m['text']}")

    return f"Context:\n{chr(10).join(snippets)}\n\nQuestion: {question}"

def answer(question, matches):
    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(question, matches)},
        ],
    )

    text = completion.choices[0].message.content.strip()

    # check against the exact marker we gave the model — no guessing
    no_info = _NO_INFO_MARKER in text.lower()

    sources = []
    if not no_info:
        for i, m in enumerate(matches):
            # only include a source if the model actually cited it as [1], [2], etc.
            citation = f"[{i + 1}]"
            if citation not in text:
                continue
            meta = m["metadata"]
            doc_type = meta.get("type", "text")
            allowed_keys = _SOURCE_KEYS.get(doc_type, _SOURCE_KEYS["text"])
            filtered = {k: meta[k] for k in allowed_keys if k in meta and meta[k] not in (None, "", "Non defini")}
            sources.append({"rank": i + 1, "score": m["score"], **filtered})

    return {"answer": text, "sources": sources}