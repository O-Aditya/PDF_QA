import os
import re
import json
import pdfplumber
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import ollama
import logging
import tempfile
import shutil

# === FLASK APP CONFIGURATION ===
app = Flask(__name__)
CORS(app, origins=["https://o-aditya.github.io", "null"])  # Allow GitHub Pages and local files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Model configuration
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:latest"
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# === GLOBAL MODELS (Load once) ===
logger.info("Loading models...")
try:
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    embedder = None


# === UTILITY FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_paths(filename):
    """Generate file paths for a given filename"""
    base_name = os.path.splitext(filename)[0]
    return {
        'pdf_path': os.path.join(UPLOAD_FOLDER, filename),
        'chunks_path': os.path.join(PROCESSED_FOLDER, f"{base_name}_chunks.jsonl"),
        'embeddings_path': os.path.join(PROCESSED_FOLDER, f"{base_name}_embeddings.npy"),
    }

# Helper: Convert table (list of lists) to Markdown

def table_to_markdown(table):
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    md = "| " + " | ".join(cell or "" for cell in header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        md += "| " + " | ".join(cell or "" for cell in row) + " |\n"
    return md

# === PROCESSING FUNCTIONS (From your original code) ===
def chunk_text_from_pdf(pdf_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Extract and chunk text and tables from PDF"""
    text = []
    table_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            table_id = 0
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text.append(page_text)
                # Extract tables
                tables = page.extract_tables()
                for t in tables:
                    # Convert table to Markdown for readability
                    if t and any(any(cell for cell in row) for row in t):
                        md = table_to_markdown(t)
                        table_chunks.append({
                            "text": md,
                            "chunk_id": f"table_{page_num}_{table_id}",
                            "type": "table",
                            "table_id": table_id,
                            "page": page_num
                        })
                        table_id += 1
        full_text = "\n".join(text)
        chunks = []
        start = 0
        while start < len(full_text):
            end = min(len(full_text), start + chunk_size)
            chunk = full_text[start:end]
            chunks.append({"text": chunk, "chunk_id": len(chunks), "type": "text"})
            start += chunk_size - overlap
        # Add table chunks at the end
        chunks.extend(table_chunks)
        return chunks
    except Exception as e:
        logger.error(f"Error chunking PDF: {e}")
        raise


def save_text_chunks(chunks, jsonl_path):
    """Save chunks to JSONL file"""
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error saving chunks: {e}")
        raise


def build_and_save_embeddings(chunks, embeddings_path):
    """Build and save embeddings for chunks"""
    try:
        texts = [c['text'] for c in chunks]
        embs = embedder.encode(texts, convert_to_numpy=True)
        np.save(embeddings_path, embs)
        return embs
    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        raise


def load_text_chunks(jsonl_path):
    """Load text chunks from JSONL file"""
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise


def rag_answer(chunks, question, embeddings, conversation_history=None):
    """Get answer using RAG pipeline with NumPy-based search and chat context"""
    try:
        q_emb = embedder.encode([question])

        # Normalize embeddings for cosine similarity calculation
        q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        embs_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine similarity and find top_k results
        similarities = np.dot(embs_norm, q_emb_norm.T).flatten()
        top_k_indices = np.argsort(similarities)[-TOP_K:][::-1]

        selected = [chunks[i]['text'] for i in top_k_indices]
        context = "\n\n".join(selected)

        # Prepare chat history for prompt
        chat_msgs = ""
        if conversation_history:
            # Only use last 6 messages for brevity
            for msg in conversation_history[-6:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    chat_msgs += f"User: {content}\n"
                else:
                    chat_msgs += f"Assistant: {content}\n"

        prompt = f"You are a helpful assistant answering questions about a PDF document.\n"
        if chat_msgs:
            prompt += f"Conversation so far:\n{chat_msgs}\n"
        prompt += f"Relevant document context:\n{context}\n\nUser question: {question}\nAssistant:"

        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        return resp['message']['content'].strip()

    except Exception as e:
        logger.error(f"Error in RAG answer: {e}")
        return f"Error processing text query: {str(e)}"


# === API ENDPOINTS ===

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": all([embedder]),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only PDF files are allowed"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"

        file_paths = get_file_paths(unique_filename)
        file.save(file_paths['pdf_path'])

        # Process PDF
        logger.info(f"Processing PDF: {unique_filename}")

        # Extract text and tables
        chunks = chunk_text_from_pdf(file_paths['pdf_path'])
        save_text_chunks(chunks, file_paths['chunks_path'])
        # Build and save embeddings
        build_and_save_embeddings(chunks, file_paths['embeddings_path'])
        num_tables = sum(1 for c in chunks if c.get('type') == 'table')
        return jsonify({
            "message": "File uploaded and processed successfully",
            "file_id": unique_filename,
            "stats": {
                "text_chunks": len([c for c in chunks if c.get('type') == 'text']),
                "tables_found": num_tables,
                "file_size": os.path.getsize(file_paths['pdf_path'])
            }
        })

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


@app.route('/api/question', methods=['POST'])
def ask_question():
    """Answer question about uploaded PDF"""
    try:
        data = request.get_json()

        if not data or 'question' not in data or 'file_id' not in data:
            return jsonify({"error": "Missing question or file_id"}), 400

        question = data['question'].strip()
        file_id = data['file_id']
        conversation_history = data.get('conversation_history', [])

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        file_paths = get_file_paths(file_id)

        # Check if processed files exist
        required_files = [file_paths['chunks_path'], file_paths['embeddings_path']]
        if not all(os.path.exists(p) for p in required_files):
            return jsonify({"error": "PDF not processed or files missing"}), 404

        # Load processed data
        chunks = load_text_chunks(file_paths['chunks_path'])
        embeddings = np.load(file_paths['embeddings_path'])

        # Text QA using RAG
        answer = rag_answer(chunks, question, embeddings, conversation_history)

        return jsonify({
            "answer": answer,
            "router": "RAG",
            "router_info": f"Text QA - {TOP_K} chunks analyzed",
            "chunks_used": TOP_K
        })

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": f"Error processing question: {str(e)}"}), 500


@app.route('/api/files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.pdf'):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file_info = {
                    "file_id": filename,
                    "original_name": filename.split('_', 2)[-1] if '_' in filename else filename,
                    "size": os.path.getsize(file_path),
                    "uploaded": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                }
                files.append(file_info)

        return jsonify({"files": files})

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"error": f"Error listing files: {str(e)}"}), 500


@app.route('/api/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete uploaded file and its processed data"""
    try:
        file_paths = get_file_paths(file_id)

        # Delete all associated files
        all_paths = [
            file_paths['pdf_path'],
            file_paths['chunks_path'],
            file_paths['embeddings_path']
        ]

        deleted_count = 0
        for path in all_paths:
            if os.path.exists(path):
                os.remove(path)
                deleted_count += 1

        return jsonify({
            "message": f"File and processed data deleted successfully",
            "files_deleted": deleted_count
        })

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"error": f"Error deleting file: {str(e)}"}), 500


@app.route('/api/sample_prompts', methods=['GET'])
def get_sample_prompts():
    """Return a list of predefined sample prompts for the UI"""
    prompts = [
        {
            "title": "📋 Document Summary",
            "desc": "Get an overview of the main topics",
            "prompt": "What is the main topic of this document?"
        },
        {
            "title": "🔍 Key Findings",
            "desc": "Extract important insights",
            "prompt": "List the key findings and conclusions"
        },
        {
            "title": "💰 Financial Data",
            "desc": "Find financial information",
            "prompt": "What are the financial highlights mentioned?"
        },
        {
            "title": "⚙️ Methodology",
            "desc": "Understand the approach taken",
            "prompt": "Summarize the methodology used"
        }
    ]
    return jsonify({"prompts": prompts})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# === MAIN ===
if __name__ == '__main__':
    logger.info("Starting PDF QA API server...")
    logger.info("Available endpoints:")
    logger.info("  GET  /api/health - Health check")
    logger.info("  POST /api/upload - Upload PDF")
    logger.info("  POST /api/question - Ask question")
    logger.info("  GET  /api/files - List files")
    logger.info("  DELETE /api/files/<file_id> - Delete file")
    logger.info("  GET  /api/sample_prompts - Get sample prompts")

    app.run(host='0.0.0.0', port=5000, debug=True)