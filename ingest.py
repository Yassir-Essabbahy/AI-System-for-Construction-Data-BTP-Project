from chunker import chunk_text
from pdf_reader import read_pdf
from embeddings import embed_documents
from vectorstore import upsert

def _ingest_chunks(chunks, metadata):
    chunks = [c for c in chunks if c and c.strip()]
    if not chunks:
        return 0
    vectors = embed_documents(chunks)
    records = [{"values": v, "metadata": {**metadata, "text": c}} for c, v in zip(chunks, vectors)]
    return upsert(records)

def ingest_text(text, metadata):
    return {"chunks_written": _ingest_chunks(chunk_text(text), metadata)}

def ingest_pdf(file_or_path, metadata):
    total = 0
    for page_num, page_text in read_pdf(file_or_path):
        total += _ingest_chunks(chunk_text(page_text), {**metadata, "page": page_num})
    return {"chunks_written": total}

def ingest_emails(emails):
    total = 0
    for email in emails:
        subject = email.get("subject") or ""
        body    = email.get("body") or ""
        text    = f"{subject}\n\n{body}".strip()
        metadata = {
            "source":    subject or "email",
            "type":      "email",
            "sender":    email.get("from") or "",
            "date":      email.get("date") or "",
            "project":   email.get("project") or "",
            "lot":       email.get("lot") or "",
            "criticite": email.get("criticite") or "",
        }
        total += _ingest_chunks(chunk_text(text), metadata)
    return {"chunks_written": total}