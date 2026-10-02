# 📚 Smart Notes – AI-Powered Academic Assistant

An AI-powered web app that transforms your PDFs into concise summaries and lets you have interactive Q&A sessions with your documents using Google Gemini, Pinecone vector search, and Supabase.

---

## ✨ Features

- 📄 **PDF Upload & Processing** – Upload PDFs, extract text automatically
- 🤖 **AI Summarization** – Google Gemini 2.5 Flash generates detailed Markdown summaries
- 💬 **Interactive Q&A** – Ask questions about your documents with RAG-based semantic search
- 🔍 **Vector Search** – Pinecone powers context-aware answers
- 📥 **Download Summaries** – Export summaries as formatted PDFs
- 🔒 **JWT Authentication** – Secure login/register with hashed passwords
- 🌙 **Dark UI** – Premium dark glassmorphism design

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + Flask |
| Database | Supabase (PostgreSQL) |
| Vector DB | Pinecone |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| LLM | Google Gemini 2.5 Flash |
| File Storage | Cloudinary |
| Auth | JWT (bcrypt passwords) |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | HTML + TailwindCSS (CDN) |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Accounts on: **Supabase**, **Cloudinary**, **Pinecone**, **HuggingFace**, **Google AI Studio**

### 2. Clone & Install

```bash
# Clone the repo (or set up from the project folder)
cd smart-notes

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install bcrypt  # for password hashing
```

### 3. Configure Environment

```bash
# Copy the example env file
copy .env.example .env      # Windows
cp .env.example .env        # macOS/Linux

# Then open .env and fill in your API keys
```

### 4. Set Up Supabase Database

1. Go to [supabase.com](https://supabase.com) and create a project
2. Open the **SQL Editor** in your Supabase dashboard
3. Paste and run the contents of `database.sql`
4. Copy your **Project URL** and **anon key** into `.env`

### 5. Set Up External Services

**Cloudinary** (`cloudinary.com`):
- Create a free account → Dashboard → copy Cloud Name, API Key, API Secret

**Pinecone** (`pinecone.io`):
- Create a free account → API Keys → copy key
- The index will be created automatically on first upload

**HuggingFace** (`huggingface.co`):
- Settings → Access Tokens → create a token with read access

**Google AI Studio** (`aistudio.google.com`):
- Get API Key → copy it

### 6. Run the App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
smart-notes/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── database.sql            # Supabase schema (run once)
├── send_email.py           # Utility: bulk magic-link emails
├── .env.example            # Environment variable template
├── .env                    # Your actual credentials (not committed)
├── .gitignore
├── templates/
│   ├── index.html          # Login / Register page
│   └── dashboard.html      # Main dashboard UI
└── static/
    └── (favicon, etc.)
```

---

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/register` | Create account |
| POST | `/api/login` | Sign in, get JWT |
| POST | `/api/change-password` | Change password (auth required) |
| GET | `/api/user` | Get current user info |

### Notes
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/notes` | List all notes |
| GET | `/api/notes/<id>` | Get note with summary |
| POST | `/api/notes/upload` | Upload PDF (multipart) |
| PATCH | `/api/notes/<id>` | Rename note |
| DELETE | `/api/notes/<id>` | Soft-delete note |
| GET | `/api/notes/<id>/download-summary` | Download summary as PDF |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Send message, get AI answer |
| GET | `/api/chat/<note_id>/history` | Get chat history |
| DELETE | `/api/chat/<note_id>/clear` | Clear chat history |

---

## 🚢 Deployment (Render)

1. Push this project to a GitHub repo
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt && pip install bcrypt`
4. Set **Start Command**: `gunicorn app:app`
5. Add all environment variables from `.env` in the Render dashboard

---

## 🔒 Security Notes

- Passwords are hashed with `bcrypt`
- JWT tokens expire after 7 days
- `.env` is in `.gitignore` – never commit it
- Supabase Row Level Security (RLS) is enabled
- Change `JWT_SECRET` and `FLASK_SECRET_KEY` to strong random values in production

---

## 📄 License

MIT – use freely for personal and commercial projects.
