from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# # ...existing code...
# REMOVE this line:
# langchain_huggingface
# ...existing code... import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangDocument
import uuid
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".")
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
VECTOR_FOLDER = "vector_store"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    """Process PDF file and create vector store"""
    try:
        logger.info(f"Processing PDF: {file_path}")
        
        # Read PDF
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        if not text.strip():
            raise ValueError("No text content found in PDF")
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        
        # Create documents
        documents = [LangDocument(page_content=chunk) for chunk in chunks]
        
        # Create and save vector store
        db = FAISS.from_documents(documents, embedding_model)
        db.save_local(VECTOR_FOLDER)
        
        logger.info(f"Successfully processed PDF with {len(documents)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Validate file presence
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File too large. Maximum size is 16MB"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Process PDF
        process_pdf(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({"message": "Book processed and indexed successfully!"})
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Failed to process file"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle question answering"""
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data.get("question").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Check if vector store exists
        if not os.path.exists(os.path.join(VECTOR_FOLDER, "index.faiss")):
            return jsonify({"answer": "No book uploaded yet. Please upload a PDF first."})
        
        # Load vector store
        db = FAISS.load_local(VECTOR_FOLDER, embedding_model, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return jsonify({"error": "GROQ API key not configured"}), 500
        
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.1
        )
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        result = qa({"query": question})
        
        return jsonify({
            "answer": result["result"],
            "source_documents": len(result.get("source_documents", []))
        })
        
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        return jsonify({"error": "Failed to process question"}), 500

@app.route("/clear", methods=["POST"])
def clear_store():
    """Clear the vector store"""
    try:
        if os.path.exists(VECTOR_FOLDER):
            shutil.rmtree(VECTOR_FOLDER)
            os.makedirs(VECTOR_FOLDER, exist_ok=True)
        
        return jsonify({"message": "Vector store cleared successfully"})
        
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return jsonify({"error": "Failed to clear store"}), 500

@app.route("/status", methods=["GET"])
def get_status():
    """Get application status"""
    has_store = os.path.exists(os.path.join(VECTOR_FOLDER, "index.faiss"))
    return jsonify({
        "has_book": has_store,
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    })

@app.route("/")
def index():
    """Serve main page"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set. The application will not work without it.")
    
    app.run(debug=True, port=5000, host="0.0.0.0")
