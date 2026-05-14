import re
import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from vectorstore import query
from embeddings import embed_query
from config import TOP_K

_client = Groq(api_key=GROQ_API_KEY)

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

# ── MULTI-QUERY RETRIEVAL ──────────────────────────────────────────────────────

def _generate_query_variants(question: str) -> list[str]:
    """Ask the LLM to rewrite the question 3 different ways."""
    prompt = (
        "Generate 3 different reformulations of this BTP construction question "
        "to improve document retrieval coverage.\n"
        "Reply with ONLY 3 lines, one reformulation per line, no numbering, no extra text.\n\n"
        f"Original question: {question}"
    )
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = response.choices[0].message.content.strip().split("\n")
        return [l.strip() for l in lines if l.strip()][:3]
    except Exception:
        return []  # graceful fallback — original question still used


def multi_query_retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieves chunks using the original question + up to 3 LLM-generated variants.
    Deduplicates by vector ID and sorts by score descending.
    Returns at most top_k * 2 unique chunks.
    """
    variants = _generate_query_variants(question)
    all_queries = [question] + variants

    seen_ids: set[str] = set()
    all_chunks: list[dict] = []

    for q in all_queries:
        embedding = embed_query(q)
        matches = query(embedding, top_k=top_k)
        for m in matches:
            vid = m.get("id") or m.get("metadata", {}).get("source", "") + str(m.get("score", 0))
            if vid in seen_ids:
                continue
            if m.get("score", 0) < 0.3:
                continue
            seen_ids.add(vid)
            all_chunks.append(m)

    all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_chunks[: top_k * 2]


# ── ANSWER GENERATION ──────────────────────────────────────────────────────────

def _build_prompt(question: str, matches: list[dict]) -> str:
    if not matches:
        return f"Context: (none)\n\nQuestion: {question}"

    snippets = []
    for i, m in enumerate(matches, start=1):
        meta = m.get("metadata", {})
        tags = ", ".join(
            f"{k}: {v}"
            for k, v in meta.items()
            if k != "type" and v not in (None, "", "Non defini")
        )
        header = f"[{i}]" + (f" ({tags})" if tags else "")
        snippets.append(f"{header}\n{m.get('text', '')}")

    return f"Context:\n{chr(10).join(snippets)}\n\nQuestion: {question}"


def answer_with_context(question: str, matches: list[dict]) -> dict:
    """
    Generates a grounded answer from retrieved chunks.
    Returns answer text + cited sources + metadata about retrieval.
    """
    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_prompt(question, matches)},
        ],
    )

    text = completion.choices[0].message.content.strip()
    no_info = _NO_INFO_MARKER in text.lower()

    sources = []
    if not no_info:
        for i, m in enumerate(matches):
            citation = f"[{i + 1}]"
            if citation not in text:
                continue
            meta = m.get("metadata", {})
            doc_type = meta.get("type", "text")
            allowed_keys = _SOURCE_KEYS.get(doc_type, _SOURCE_KEYS["text"])
            filtered = {
                k: meta[k]
                for k in allowed_keys
                if k in meta and meta[k] not in (None, "", "Non defini")
            }
            sources.append({"rank": i + 1, "score": m.get("score"), **filtered})

    return {
        "answer": text,
        "sources": sources,
        "chunks_retrieved": len(matches),
        "queries_used": 4,
    }


# ── LEGACY WRAPPER (keeps backward compatibility) ──────────────────────────────

def answer(question: str, matches: list[dict]) -> dict:
    """
    Backward-compatible wrapper — still used if /ask is called
    with pre-fetched matches. Prefer answer_with_context().
    """
    return answer_with_context(question, matches)


# ── COMPLIANCE ANALYSIS ────────────────────────────────────────────────────────

def analyze_compliance(text: str, project: str = "Non spécifié") -> dict:
    """
    Analyses a BTP text for regulatory risks (DTU, NF/EN/ISO norms)
    and site risks. Returns structured JSON with criticality level,
    risks list, and recommended actions.
    """
    prompt = f"""Tu es un expert BTP spécialisé en conformité réglementaire française.
Analyse le texte suivant extrait d'un projet BTP et identifie :

1. Risques réglementaires : non-conformités potentielles avec les DTU, normes NF/EN/ISO
2. Risques chantier : problèmes de sécurité, délais, ou qualité détectés
3. Actions recommandées : corrections prioritaires à apporter
4. Niveau de criticité global : FAIBLE / MOYEN / ÉLEVÉ / CRITIQUE

Projet : {project}

Texte à analyser :
{text[:3000]}

Réponds UNIQUEMENT en JSON valide avec cette structure exacte, sans texte avant ou après :
{{
  "criticite": "FAIBLE|MOYEN|ÉLEVÉ|CRITIQUE",
  "risques_reglementaires": ["..."],
  "risques_chantier": ["..."],
  "actions_recommandees": ["..."],
  "resume": "..."
}}"""

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            analysis = json.loads(json_match.group())
        except json.JSONDecodeError:
            analysis = {"resume": raw, "criticite": "INCONNU"}
    else:
        analysis = {"resume": raw, "criticite": "INCONNU"}

    return {
        "project": project,
        "analysis": analysis,
        "model": GROQ_MODEL,
        "status": "success",
        "text_length": len(text),
    }