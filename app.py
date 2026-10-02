"""
Smart Notes - AI-Powered Academic Assistant
Flask backend with PDF processing, vector search, and Gemini AI.
"""

import os
import uuid
import time
import logging
import warnings
import markdown
from io import BytesIO
from typing import List
from datetime import datetime, timedelta, timezone
from functools import wraps
from threading import Timer

import jwt
import pymupdf as fitz  # PyMuPDF (updated API)
import httpx
from retry import retry
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*")

# ── Flask app ─────────────────────────────────────────────────────────────────
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
CORS(app)

# ── Environment variables ─────────────────────────────────────────────────────
SUPABASE_URL           = os.getenv("SUPABASE_URL")
# Use service role key to bypass RLS for server-side operations
SUPABASE_KEY           = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY")
CLOUDINARY_CLOUD_NAME  = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY     = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET  = os.getenv("CLOUDINARY_API_SECRET")
JWT_SECRET             = os.getenv("JWT_SECRET", "jwt-secret-change-me")
PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "smart-notes-index")
HUGGINGFACE_API_KEY    = os.getenv("HUGGINGFACE_API_KEY")

_required = [
    SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,
    PINECONE_API_KEY, HUGGINGFACE_API_KEY,
]
if not all(_required):
    logger.warning("One or more environment variables are missing – some features may not work.")
else:
    logger.info("All environment variables loaded.")


# ── Lazy service initializer ──────────────────────────────────────────────────
class LazyServices:
    """Deferred initialization of heavy AI/cloud services."""

    def __init__(self):
        self._supabase       = None
        self._cloudinary     = None
        self._pinecone_index = None
        self._hf_embeddings  = None
        self._llm            = None
        self._nltk_ready     = False

    # Supabase ----------------------------------------------------------------
    def supabase(self):
        if self._supabase is None:
            from supabase import create_client
            self._supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase initialized")
        return self._supabase

    # Cloudinary --------------------------------------------------------------
    def cloudinary(self):
        if self._cloudinary is None:
            import cloudinary
            import cloudinary.uploader
            c_url = os.getenv("CLOUDINARY_URL")
            c_name = (CLOUDINARY_CLOUD_NAME or "").strip()
            c_key = (CLOUDINARY_API_KEY or "").strip()
            c_secret = (CLOUDINARY_API_SECRET or "").strip()

            if c_url or c_name.startswith("cloudinary://"):
                cloudinary.config(cloudinary_url=c_url or c_name)
            else:
                cloudinary.config(
                    cloud_name=c_name,
                    api_key=c_key,
                    api_secret=c_secret,
                )
            self._cloudinary = cloudinary
            logger.info("Cloudinary initialized")
        return self._cloudinary

    # Pinecone ----------------------------------------------------------------
    def pinecone_index(self):
        if self._pinecone_index is None:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [idx["name"] for idx in pc.list_indexes()]
            if PINECONE_INDEX_NAME not in existing:
                logger.info("Creating Pinecone index …")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                time.sleep(10)
            self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            logger.info("Pinecone index ready")
        return self._pinecone_index

    # HuggingFace embeddings --------------------------------------------------
    def hf_embeddings(self):
        if self._hf_embeddings is None:
            from huggingface_hub import InferenceClient

            class _HFEmbed:
                MODEL = "sentence-transformers/all-MiniLM-L6-v2"

                def __init__(self):
                    self.client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

                @retry(httpx.HTTPError, tries=3, delay=2, backoff=2)
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    if not texts:
                        return []
                    resp = self.client.feature_extraction(texts, model=self.MODEL)
                    return resp.tolist()

                def embed_query(self, text: str) -> List[float]:
                    result = self.embed_documents([text])
                    return result[0] if result else []

            self._hf_embeddings = _HFEmbed()
            logger.info("HuggingFace embeddings initialized")
        return self._hf_embeddings

    # Gemini LLM --------------------------------------------------------------
    def llm(self):
        if self._llm is None:
            import google.generativeai as genai
            from langchain_google_genai import GoogleGenerativeAI
            genai.configure(api_key=GOOGLE_API_KEY)
            self._llm = GoogleGenerativeAI(
                model="gemini-3.6-flash", google_api_key=GOOGLE_API_KEY
            )
            logger.info("Google Gemini LLM initialized")
        return self._llm

    # NLTK --------------------------------------------------------------------
    def ensure_nltk(self):
        if not self._nltk_ready:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            self._nltk_ready = True


services = LazyServices()


# ── Timeout helper ────────────────────────────────────────────────────────────
def timeout(seconds):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t = Timer(seconds, lambda: None)  # Windows-safe no-op timer
            t.start()
            try:
                return fn(*args, **kwargs)
            finally:
                t.cancel()
        return wrapper
    return decorator


# ── Rate-limited LLM call ────────────────────────────────────────────────────
def safe_llm_call(func, *args, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as exc:
            err = str(exc)
            if any(k in err for k in ("429", "quota", "ResourceExhausted")):
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited – retrying in {wait}s")
                    time.sleep(wait)
                    continue
            logger.error(f"LLM error: {exc}")
            return None
    return None


# ── Auth helpers ──────────────────────────────────────────────────────────────
def generate_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        logger.warning("JWT expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT: {e}")
    return None


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        if not token:
            return jsonify({"error": "No token provided"}), 401
        payload = verify_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        request.current_user = payload
        return fn(*args, **kwargs)
    return wrapper


# ── PDF utilities ─────────────────────────────────────────────────────────────
@timeout(60)
def extract_pdf_text(file_bytes: bytes) -> str | None:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        logger.error(f"PDF extraction error: {exc}")
        return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def store_embeddings(namespace: str, chunks: List[str]) -> bool:
    try:
        embedder = services.hf_embeddings()
        index    = services.pinecone_index()
        vectors  = []
        batch    = 50
        for i in range(0, len(chunks), batch):
            batch_chunks = chunks[i : i + batch]
            embeds = embedder.embed_documents(batch_chunks)
            for j, (chunk, emb) in enumerate(zip(batch_chunks, embeds)):
                vectors.append({
                    "id": f"{namespace}-{i+j}",
                    "values": emb,
                    "metadata": {"text": chunk, "namespace": namespace},
                })
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info(f"Stored {len(vectors)} vectors for namespace '{namespace}'")
        return True
    except Exception as exc:
        logger.error(f"Embedding storage error: {exc}")
        return False


def semantic_search(namespace: str, query: str, top_k: int = 5) -> List[str]:
    try:
        embedder = services.hf_embeddings()
        index    = services.pinecone_index()
        q_emb    = embedder.embed_query(query)
        results  = index.query(
            vector=q_emb, top_k=top_k, namespace=namespace, include_metadata=True
        )
        return [m["metadata"]["text"] for m in results.get("matches", [])]
    except Exception as exc:
        logger.error(f"Semantic search error: {exc}")
        return []


# ── Markdown → HTML ───────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    return html


# ── PDF generation from summary ───────────────────────────────────────────────
def build_pdf_from_summary(title: str, summary_html: str) -> bytes:
    buf    = BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter, rightMargin=54, leftMargin=54,
                               topMargin=72, bottomMargin=54)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body2", parent=styles["Normal"], fontSize=11,
                               leading=16, alignment=TA_JUSTIFY, spaceAfter=8))
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20,
                               textColor=colors.HexColor("#3c55bf"), spaceAfter=12))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15,
                               textColor=colors.HexColor("#5566cc"), spaceAfter=8))

    soup    = BeautifulSoup(summary_html, "html.parser")
    story   = [Paragraph(title, styles["H1"]), Spacer(1, 12)]

    for el in soup.children:
        tag = getattr(el, "name", None)
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if tag in ("h1", "h2", "h3"):
            story.append(Paragraph(txt, styles["H2"]))
        elif tag == "ul":
            items = [ListItem(Paragraph(li.get_text(" ", strip=True), styles["Body2"]))
                     for li in el.find_all("li")]
            story.append(ListFlowable(items, bulletType="bullet"))
        elif tag == "ol":
            items = [ListItem(Paragraph(li.get_text(" ", strip=True), styles["Body2"]))
                     for li in el.find_all("li")]
            story.append(ListFlowable(items, bulletType="1"))
        elif tag == "pre":
            story.append(Preformatted(txt, styles["Code"]))
        else:
            story.append(Paragraph(txt, styles["Body2"]))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# ── Routes ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Frontend pages
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    db = services.supabase()
    try:
        existing = db.table("users").select("id").eq("email", email).execute()
        if existing.data:
            return jsonify({"error": "Email already registered"}), 409

        import bcrypt
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        result = db.table("users").insert({
            "email": email,
            "password": hashed,
            "tier": "free",
        }).execute()

        user  = result.data[0]
        token = generate_token(user["id"], user["email"])
        return jsonify({"token": token, "user": {"id": user["id"], "email": email, "tier": "free"}}), 201

    except Exception as exc:
        logger.error(f"Register error: {exc}")
        return jsonify({"error": "Registration failed"}), 500


@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = services.supabase()
    try:
        result = db.table("users").select("*").eq("email", email).execute()
        if not result.data:
            return jsonify({"error": "Invalid credentials"}), 401

        user = result.data[0]
        import bcrypt
        if not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"error": "Invalid credentials"}), 401

        token = generate_token(user["id"], user["email"])
        return jsonify({
            "token": token,
            "user": {"id": user["id"], "email": user["email"], "tier": user.get("tier", "free")},
        })

    except Exception as exc:
        logger.error(f"Login error: {exc}")
        return jsonify({"error": "Login failed"}), 500


@app.route("/api/change-password", methods=["POST"])
@require_auth
def change_password():
    data         = request.get_json() or {}
    old_password = (data.get("old_password") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not old_password or not new_password:
        return jsonify({"error": "Both passwords required"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400

    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = db.table("users").select("password").eq("id", user_id).execute()
        if not result.data:
            return jsonify({"error": "User not found"}), 404

        import bcrypt
        if not bcrypt.checkpw(old_password.encode(), result.data[0]["password"].encode()):
            return jsonify({"error": "Old password is incorrect"}), 401

        new_hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        db.table("users").update({"password": new_hashed}).eq("id", user_id).execute()
        return jsonify({"message": "Password changed successfully"})

    except Exception as exc:
        logger.error(f"Change password error: {exc}")
        return jsonify({"error": "Failed to change password"}), 500


@app.route("/api/user", methods=["GET"])
@require_auth
def get_user():
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = db.table("users").select("id, email, tier, created_at").eq("id", user_id).execute()
        if not result.data:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"user": result.data[0]})
    except Exception as exc:
        logger.error(f"Get user error: {exc}")
        return jsonify({"error": "Failed to fetch user"}), 500


# ── Notes ─────────────────────────────────────────────────────────────────────
@app.route("/api/notes", methods=["GET"])
@require_auth
def get_notes():
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = (
            db.table("notes")
            .select("id, title, file_type, file_url, created_at, updated_at")
            .eq("user_id", user_id)
            .eq("deleted", False)
            .order("created_at", desc=True)
            .execute()
        )
        return jsonify({"notes": result.data or []})
    except Exception as exc:
        logger.error(f"Get notes error: {exc}")
        return jsonify({"error": "Failed to fetch notes"}), 500


@app.route("/api/notes/<note_id>", methods=["GET"])
@require_auth
def get_note(note_id):
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = (
            db.table("notes")
            .select("*")
            .eq("id", note_id)
            .eq("user_id", user_id)
            .eq("deleted", False)
            .execute()
        )
        if not result.data:
            return jsonify({"error": "Note not found"}), 404
        note = result.data[0]
        if note.get("summary"):
            note["summary_html"] = md_to_html(note["summary"])
        return jsonify({"note": note})
    except Exception as exc:
        logger.error(f"Get note error: {exc}")
        return jsonify({"error": "Failed to fetch note"}), 500


@app.route("/api/notes/upload", methods=["POST"])
@require_auth
def upload_note():
    user_id = request.current_user["user_id"]

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    file_bytes = file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB cap
        return jsonify({"error": "File too large (max 10 MB)"}), 400

    # 1. Upload to Cloudinary
    try:
        cl = services.cloudinary()
        import cloudinary.uploader
        upload_result = cloudinary.uploader.upload(
            file_bytes,
            resource_type="raw",
            folder="smart-notes",
            public_id=f"{user_id}/{uuid.uuid4()}",
        )
        file_url = upload_result.get("secure_url")
    except Exception as exc:
        logger.error(f"Cloudinary upload error: {exc}")
        return jsonify({"error": "File upload failed"}), 500

    # 2. Extract text
    text = extract_pdf_text(file_bytes)
    if not text or len(text.strip()) < 50:
        return jsonify({"error": "Could not extract text from PDF"}), 422

    # 3. Chunk + embed
    chunks    = chunk_text(text)
    namespace = f"user-{user_id}-{uuid.uuid4().hex[:8]}"
    store_embeddings(namespace, chunks)

    # 4. Generate summary via Gemini
    llm   = services.llm()
    title = (request.form.get("title") or file.filename.rsplit(".", 1)[0] or "Untitled Note")

    prompt = (
        f"You are an expert academic summarizer. Read the following document content carefully "
        f"and produce a detailed, well-structured summary in Markdown format. "
        f"Include: an overview, key concepts, main points (with bullet lists), and a conclusion.\n\n"
        f"Document title: {title}\n\n"
        f"Content (first 8000 chars):\n{text[:8000]}"
    )
    summary_md = safe_llm_call(llm.invoke, prompt) or "Summary could not be generated."

    # 5. Save note to Supabase
    db = services.supabase()
    try:
        result = db.table("notes").insert({
            "user_id":            user_id,
            "title":              title,
            "content":            text[:20000],
            "summary":            summary_md,
            "file_url":           file_url,
            "file_type":          "pdf",
            "pinecone_namespace": namespace,
        }).execute()
        note = result.data[0]
        note["summary_html"] = md_to_html(summary_md)
        return jsonify({"note": note}), 201
    except Exception as exc:
        logger.error(f"Note save error: {exc}")
        return jsonify({"error": "Failed to save note"}), 500


@app.route("/api/notes/<note_id>", methods=["PATCH"])
@require_auth
def rename_note(note_id):
    user_id = request.current_user["user_id"]
    data    = request.get_json() or {}
    title   = (data.get("title") or "").strip()

    if not title:
        return jsonify({"error": "Title is required"}), 400

    db = services.supabase()
    try:
        result = (
            db.table("notes")
            .update({"title": title, "updated_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", note_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not result.data:
            return jsonify({"error": "Note not found"}), 404
        return jsonify({"note": result.data[0]})
    except Exception as exc:
        logger.error(f"Rename error: {exc}")
        return jsonify({"error": "Failed to rename note"}), 500


@app.route("/api/notes/<note_id>", methods=["DELETE"])
@require_auth
def delete_note(note_id):
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        # Soft delete
        db.table("notes").update({"deleted": True}).eq("id", note_id).eq("user_id", user_id).execute()
        # Also delete chat history for this note
        db.table("chats").delete().eq("note_id", note_id).eq("user_id", user_id).execute()
        return jsonify({"message": "Note deleted"})
    except Exception as exc:
        logger.error(f"Delete note error: {exc}")
        return jsonify({"error": "Failed to delete note"}), 500


@app.route("/api/notes/<note_id>/download-summary", methods=["GET"])
@require_auth
def download_summary(note_id):
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = (
            db.table("notes")
            .select("title, summary")
            .eq("id", note_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not result.data or not result.data[0].get("summary"):
            return jsonify({"error": "Summary not found"}), 404

        note        = result.data[0]
        summary_html = md_to_html(note["summary"])
        pdf_bytes   = build_pdf_from_summary(note["title"], summary_html)

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{note["title"]}_summary.pdf"'},
        )
    except Exception as exc:
        logger.error(f"Download summary error: {exc}")
        return jsonify({"error": "Failed to generate PDF"}), 500


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    user_id = request.current_user["user_id"]
    data    = request.get_json() or {}
    note_id = data.get("note_id")
    message = (data.get("message") or "").strip()

    if not note_id or not message:
        return jsonify({"error": "note_id and message are required"}), 400

    db = services.supabase()
    try:
        note_res = (
            db.table("notes")
            .select("title, pinecone_namespace, content")
            .eq("id", note_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not note_res.data:
            return jsonify({"error": "Note not found"}), 404

        note      = note_res.data[0]
        namespace = note.get("pinecone_namespace")

        # Retrieve relevant context
        context_chunks = semantic_search(namespace, message) if namespace else []
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else note.get("content", "")[:4000]

        # Fetch recent chat history for context
        history_res = (
            db.table("chats")
            .select("user_message, ai_response")
            .eq("note_id", note_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        history = list(reversed(history_res.data or []))
        history_text = ""
        for h in history:
            history_text += f"User: {h['user_message']}\nAssistant: {h['ai_response']}\n\n"

        prompt = (
            f"You are an intelligent academic assistant helping a student understand their document.\n"
            f"Document title: {note['title']}\n\n"
            f"Relevant excerpts from the document:\n{context}\n\n"
            f"Previous conversation:\n{history_text}"
            f"Student question: {message}\n\n"
            f"Answer the question thoroughly using the document content. "
            f"Format your answer in Markdown with clear structure."
        )

        llm      = services.llm()
        response = safe_llm_call(llm.invoke, prompt)
        if not response:
            return jsonify({"error": "AI response failed. Please try again."}), 503

        # Save to DB
        db.table("chats").insert({
            "user_id":      user_id,
            "note_id":      note_id,
            "user_message": message,
            "ai_response":  response,
        }).execute()

        return jsonify({
            "response":      response,
            "response_html": md_to_html(response),
        })

    except Exception as exc:
        logger.error(f"Chat error: {exc}")
        return jsonify({"error": "Chat request failed"}), 500


@app.route("/api/chat/<note_id>/history", methods=["GET"])
@require_auth
def chat_history(note_id):
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        result = (
            db.table("chats")
            .select("id, user_message, ai_response, created_at")
            .eq("note_id", note_id)
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
        chats = result.data or []
        for c in chats:
            c["ai_response_html"] = md_to_html(c["ai_response"])
        return jsonify({"chats": chats})
    except Exception as exc:
        logger.error(f"Chat history error: {exc}")
        return jsonify({"error": "Failed to fetch chat history"}), 500


@app.route("/api/chat/<note_id>/clear", methods=["DELETE"])
@require_auth
def clear_chat(note_id):
    user_id = request.current_user["user_id"]
    db      = services.supabase()
    try:
        db.table("chats").delete().eq("note_id", note_id).eq("user_id", user_id).execute()
        return jsonify({"message": "Chat cleared"})
    except Exception as exc:
        logger.error(f"Clear chat error: {exc}")
        return jsonify({"error": "Failed to clear chat"}), 500


# ── Static / misc ─────────────────────────────────────────────────────────────
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "Smart Notes",
        "short_name": "SmartNotes",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#121222",
        "theme_color": "#b9c3ff",
        "icons": [],
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"Starting Smart Notes on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
