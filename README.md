# BTP AI — RAG Pipeline for Construction Documents

A retrieval-augmented generation (RAG) system built for construction companies. Upload PDFs and emails, ask questions in natural language, and get answers grounded strictly in your documents — with source citations.

**Stack:** Flask · Pinecone · Groq (LLaMA 3) · Pinecone Inference Embeddings · Vanilla HTML dashboard

---

## Project Structure

```
.
├── app.py              # Flask API — all endpoints
├── config.py           # Environment variables and constants
├── ingest.py           # Orchestrates chunking + embedding + upsert
├── chunker.py          # Splits text into overlapping chunks
├── embeddings.py       # Pinecone inference embeddings (batched)
├── vectorstore.py      # Pinecone upsert / query / clear
├── llm.py              # Groq LLM call + prompt builder + source filtering
├── pdf_reader.py       # Extracts text page by page from PDFs
├── dashboard.html      # Frontend — open directly in a browser
└── .env                # Your secrets (never commit this)
```

---

## Prerequisites

- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) account (free tier works)
- A [Groq](https://console.groq.com/) account (free tier works)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/btp-ai.git
cd btp-ai
```

**2. Install dependencies**
```bash
pip install flask flask-cors pinecone groq pypdf python-dotenv
```

**3. Create your `.env` file**

Create a file called `.env` at the root of the project:
```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional — defaults shown below
PINECONE_INDEX_NAME=btp-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBEDDING_MODEL=multilingual-e5-large
GROQ_MODEL=llama-3.1-8b-instant
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
MIN_SCORE=0.5
```

> `multilingual-e5-large` is a Pinecone-hosted embedding model — no separate API key needed, it uses your Pinecone key.

**4. Start the Flask server**
```bash
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
```

**5. Open the dashboard**

Just open `dashboard.html` directly in your browser — no server needed for the frontend.

---

## Testing the API

You can test every endpoint with `curl` or any REST client (Postman, Insomnia, etc.).

### Health check
```bash
curl http://localhost:5000/health
```
Expected:
```json
{ "status": "ok" }
```

---

### Upload a PDF
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@Asphalt.pdf" \
  -F "project=Chantier A" \
  -F "lot=VRD" \
  -F "criticite=Haute"
```
Expected:
```json
{ "message": "PDF ingested.", "chunks_written": 3 }
```
`chunks_written` will vary depending on the size of your PDF.

---

### Ingest emails
```bash
curl -X POST http://localhost:5000/ingest-emails \
  -H "Content-Type: application/json" \
  -d @emails.json_body.json
```

Where `emails.json_body.json` contains:
```json
{
  "emails": [
    {
      "subject": "Project delay update",
      "from": "client@btp.com",
      "date": "2026-05-01",
      "body": "The project deadline has been extended to June due to material delays.",
      "project": "Chantier A",
      "lot": "Gros Oeuvre",
      "criticite": "Haute"
    }
  ]
}
```
Expected:
```json
{ "message": "Emails ingested.", "chunks_written": 1 }
```

---

### Ingest plain text
```bash
curl -X POST http://localhost:5000/ingest-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitumen is a sticky black material derived from crude oil.",
    "source": "internal-note",
    "project": "Chantier B",
    "lot": "VRD"
  }'
```
Expected:
```json
{ "message": "Text ingested.", "chunks_written": 1 }
```

---

### Ask a question
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{ "question": "What is asphalt made of?" }'
```
Expected (example):
```json
{
  "question": "What is asphalt made of?",
  "answer": "Asphalt is made of bitumen and aggregates [1]. It is used mainly for road construction and provides flexibility and durability [1].",
  "sources": [
    {
      "rank": 1,
      "score": 0.91,
      "source": "Asphalt.pdf",
      "type": "pdf",
      "page": 1,
      "project": "Chantier A"
    }
  ]
}
```

If nothing relevant is found, the model replies:
```json
{
  "answer": "I don't have enough information in the provided documents.",
  "sources": []
}
```

---

### Clear the index
```bash
curl -X POST http://localhost:5000/clear
```
Expected:
```json
{ "message": "Index cleared." }
```
> ⚠️ This wipes everything from Pinecone. Use with care.

---

## What to Expect

| Scenario | What happens |
|---|---|
| Question matches a document | Answer with `[1]`, `[2]` citations + source tags |
| Question is off-topic | `"I don't have enough information in the provided documents."` |
| Same PDF page retrieved multiple times | Deduplicated — shown only once in source tags |
| PDF has multiple pages | Each page is chunked and indexed separately |
| Large PDF (50+ pages) | Works — embeddings are batched internally in groups of 96 |
| Multilingual question | The model replies in the same language as the question |

---

## How It Works

```
User question
     │
     ▼
embed_query()          — converts the question to a 1024-dim vector
     │
     ▼
vectorstore.query()    — finds the top 5 most similar chunks in Pinecone (MIN_SCORE ≥ 0.5)
     │
     ▼
llm.answer()           — sends chunks as context to LLaMA 3 via Groq
     │
     ▼
source filtering       — only sources cited as [1][2]… by the model are returned
     │
     ▼
JSON response          — answer + deduplicated sources
```

Chunking uses a sliding window: each chunk is up to 500 characters, with 100 characters of overlap carried into the next chunk so context isn't lost at boundaries.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Swap to `llama-3.3-70b-versatile` for better quality |
| `TOP_K` | `5` | How many chunks to retrieve per question |
| `MIN_SCORE` | `0.5` | Minimum cosine similarity to include a chunk |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters of overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `multilingual-e5-large` | Pinecone-hosted model, supports French + Arabic + English |

---

## Notes

- The model is instructed to use **only** the provided documents. It will not answer from general knowledge.
- The dashboard stats (document count, question count, latency) are session-only — they reset on page reload.
- The Pinecone free tier supports one index with up to 100k vectors, which is enough for hundreds of PDFs.
"# AI-System-for-Construction-Data-BTP-Project" 
