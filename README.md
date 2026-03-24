# Jesse Livermore RAG ChatBot

> Retrieval-Augmented Generation chatbot trained on 1,800 Q&A pairs from
> *Reminiscences of a Stock Operator*

---

## Repo Structure

```
livermore_chatbot/
├── backend/                        ← Deploy to Render
│   ├── app.py                      ← Flask RAG API
│   ├── requirements.txt
│   ├── render.yaml
│   └── Livermore_QA_Dataset_Extended.csv   ← ADD THIS FILE
│
└── frontend/                       ← Deploy to Netlify
    ├── index.html                  ← Full chat UI
    └── netlify.toml
```

---

## Step-by-step Deployment

### 1. Add your CSV to backend/

Copy your dataset into the backend folder:
```
backend/Livermore_QA_Dataset_Extended.csv
```

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Jesse Livermore ChatBot"
git remote add origin https://github.com/YOUR_USERNAME/livermore-chatbot.git
git push -u origin main
```

### 3. Deploy Backend → Render (free)

1. Go to **https://render.com** → Sign up with GitHub
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Set these settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Deploy**
6. Wait ~5 minutes for first deploy (embedding model downloads)
7. Copy your Render URL → looks like `https://livermore-chatbot-api.onrender.com`

> ⚠️ Free Render instances sleep after 15 minutes of inactivity.
> First request after sleep takes ~30 seconds to wake up. This is normal.

### 4. Deploy Frontend → Netlify (free)

1. Go to **https://netlify.com** → Sign up with GitHub
2. Click **Add new site → Import an existing project**
3. Connect your GitHub repo
4. Set:
   - **Base directory:** `frontend`
   - **Publish directory:** `frontend`
   - Leave build command empty
5. Click **Deploy**
6. Your site is live at `https://your-site-name.netlify.app`

### 5. Use the ChatBot

1. Open your Netlify URL
2. Paste your **Render backend URL** in the top field
3. Paste your **Groq API key** (free at https://console.groq.com)
4. Ask questions!

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/paraphrase-MiniLM-L6-v2` |
| Vector DB | FAISS |
| RAG Framework | LangChain |
| LLM | Llama 3.3 70B via Groq |
| Backend | Flask + Gunicorn |
| Backend hosting | Render (free tier) |
| Frontend | Vanilla HTML/CSS/JS |
| Frontend hosting | Netlify (free tier) |

---

## Local Development

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5000

# Open frontend/index.html in browser
# Set backend URL to http://localhost:5000
```
