from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buffer = [], ""

    for p in paragraphs:
        if len(buffer) + len(p) + 1 <= chunk_size:
            buffer = f"{buffer} {p}".strip()
        else:
            if buffer:
                chunks.append(buffer)
                tail = buffer[-overlap:] if overlap else ""
            else:
                tail = ""

            if len(p) > chunk_size:
                stride = max(1, chunk_size - overlap)
                chunks += [p[i:i + chunk_size] for i in range(0, len(p), stride)]
                buffer = ""
            else:
                buffer = f"{tail} {p}".strip() if tail else p

    if buffer:
        chunks.append(buffer)

    return chunks