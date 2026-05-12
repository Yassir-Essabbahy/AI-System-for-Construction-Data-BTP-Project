from flask import Flask, request, jsonify
from flask_cors import CORS
from config import TOP_K
from vectorstore import ensure_index, query, clear
from embeddings import embed_query
from llm import answer
from ingest import ingest_text, ingest_pdf, ingest_emails
from flask import Flask, request, jsonify, send_from_directory

@app.get("/")
def index():
    return send_from_directory(".", "dashboard.html")

app = Flask(__name__)
CORS(app)
ensure_index()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400
    try:
        matches = query(embed_query(question), top_k=TOP_K)
        return jsonify({"question": question, **answer(question, matches)})
    except Exception as e:
        app.logger.exception("ask failed")
        return jsonify({"error": str(e)}), 500

@app.post("/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (expected field 'file')."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400
    metadata = {
        "source": f.filename, "type": "pdf",
        "project": request.form.get("project", ""),
        "lot": request.form.get("lot", ""),
        "criticite": request.form.get("criticite", ""),
    }
    try:
        return jsonify({"message": "PDF ingested.", **ingest_pdf(f.stream, metadata)})
    except Exception as e:
        app.logger.exception("upload failed")
        return jsonify({"error": str(e)}), 500

@app.post("/ingest-text")
def ingest_text_route():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    metadata = {
        "source": data.get("source", "manual"), "type": "text",
        "project": data.get("project", ""),
        "lot": data.get("lot", ""),
        "criticite": data.get("criticite", ""),
    }
    try:
        return jsonify({"message": "Text ingested.", **ingest_text(text, metadata)})
    except Exception as e:
        app.logger.exception("ingest-text failed")
        return jsonify({"error": str(e)}), 500

@app.post("/ingest-emails")
def ingest_emails_route():
    data = request.get_json(silent=True) or {}
    emails = data.get("emails", [])
    if not emails:
        return jsonify({"error": "Field 'emails' is required and must be a list."}), 400
    try:
        return jsonify({"message": "Emails ingested.", **ingest_emails(emails)})
    except Exception as e:
        app.logger.exception("ingest-emails failed")
        return jsonify({"error": str(e)}), 500

@app.post("/clear")
def clear_route():
    try:
        clear()
        return jsonify({"message": "Index cleared."})
    except Exception as e:
        app.logger.exception("clear failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)