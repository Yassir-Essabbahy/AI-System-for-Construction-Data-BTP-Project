from pypdf import PdfReader

def read_pdf(source):
    reader = PdfReader(source)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages