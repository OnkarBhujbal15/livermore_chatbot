"""
Jesse Livermore RAG ChatBot — Backend (Render)
===============================================
Deployed to: https://your-app.onrender.com
"""

import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "Livermore_QA_Dataset_Extended.csv"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
TOP_K       = 3
PORT        = int(os.environ.get("PORT", 5000))

app = Flask(__name__)

# Allow requests from Netlify frontend + localhost for dev
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://*.netlify.app",      # your Netlify domain
    "https://*.netlify.com",
])

# ── Build vector DB once at startup ──────────────────────────────────────────
print("Loading dataset...")
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

print("Building embeddings (this takes ~60s on first run)...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_db       = FAISS.from_documents(chunks, embedding_model)

print(f"Ready — {len(chunks)} chunks from {len(documents)} Q&A pairs.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Livermore ChatBot API is running."})


@app.route("/ask", methods=["POST"])
def ask():
    data    = request.get_json()
    query   = (data.get("query") or "").strip()
    # Use env variable if set, otherwise accept from request body
    api_key = os.environ.get("GROQ_API_KEY") or (data.get("api_key") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400
    if not api_key:
        return jsonify({"error": "Groq API key required"}), 400

    # Retrieve relevant context from vector DB
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


@app.route("/stats")
def stats():
    return jsonify({
        "total_pairs": len(df),
        "labels":      df["Label"].value_counts().to_dict(),
        "chunks":      len(chunks),
        "model":       GROQ_MODEL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
