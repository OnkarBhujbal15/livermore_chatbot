"""
Jesse Livermore RAG ChatBot — Flask Backend
Deploy on Render as a Web Service.
Start command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "Livermore_QA_Dataset_Extended.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
FAISS_PATH  = "faiss_index"
TOP_K       = 3
PORT        = int(os.environ.get("PORT", 7860))

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])

# ── Load embedding model ──────────────────────────────────────────────────────
print("Loading embedding model...", flush=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ── Load or build FAISS index ─────────────────────────────────────────────────
if os.path.exists(FAISS_PATH):
    print("Loading saved FAISS index...", flush=True)
    vector_db = FAISS.load_local(
        FAISS_PATH, embedding_model,
        allow_dangerous_deserialization=True
    )
    df = pd.read_csv(CSV_PATH)
    print(f"Index loaded — {len(df)} Q&A pairs.", flush=True)
else:
    print("Building FAISS index for first time...", flush=True)
    df = pd.read_csv(CSV_PATH)
    documents = [
        Document(
            page_content=row["Answers"],
            metadata={"label": row["Label"], "question": row["Questions"]}
        )
        for _, row in df.iterrows()
    ]
    splitter  = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks    = splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local(FAISS_PATH)
    print(f"Index built and saved — {len(chunks)} chunks.", flush=True)

print("Flask app ready — routes active.", flush=True)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Livermore ChatBot API is running."})


@app.route("/stats")
def stats():
    return jsonify({
        "total_pairs": len(df),
        "labels":      df["Label"].value_counts().to_dict(),
        "chunks":      vector_db.index.ntotal,
        "model":       GROQ_MODEL,
        "embed_model": EMBED_MODEL,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Use env variable (set on Render), fallback to request body for local dev
    api_key = os.environ.get("GROQ_API_KEY") or (data.get("api_key") or "").strip()

    if not api_key:
        return jsonify({"error": "Groq API key not configured on server."}), 400

    # Retrieve relevant context
    retrieved = vector_db.similarity_search(query, k=TOP_K)
    context   = "\n\n".join([doc.page_content for doc in retrieved])
    labels    = list({doc.metadata.get("label", "") for doc in retrieved})

    # Build prompt
    prompt = f"""You are a Jesse Livermore trading expert chatbot trained on
"Reminiscences of a Stock Operator". Answer using only the context below.
Speak in first person as Livermore. Be concise, practical, and direct.
If context is insufficient, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

    # Call Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )

    return jsonify({
        "answer":  response.choices[0].message.content.strip(),
        "context": context,
        "labels":  labels,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
