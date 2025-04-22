import os
import fitz  # PyMuPDF for PDF text extraction
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to extract text from PDFs
def extract_text(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            logging.error(f"File not found: {pdf_path}")
            return ""
        
        doc = fitz.open(pdf_path)
        text = [page.get_text("text") for page in doc]
        return "\n".join(text)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Define the directory containing PDFs
content_dir = r"C:\Users\LENOVO\Desktop\RAG_AI_Interviewer-main\RAG_AI_Interviewer-main\docs"
if not os.path.exists(content_dir):
    logging.error(f"Directory not found: {content_dir}")
    exit()

# Define topic mapping for each PDF file
topic_mapping = {
    "cybersecurity-all-in-one-for-dummies-9781394152872-9781394152858-9781394152865.pdf": "cybersecurity",
    "Hands On Machine Learning with Scikit Learn and TensorFlow.pdf": "machine_learning",
    "Mastering machine learning algorithms  expert techniques for implementing popular machine learning algorithms, fine-tuning... (Giuseppe Bonaccorso) (z-lib.org).pdf": "machine_learning",
    "The Data Warehouse Toolkit - Kimball.pdf": "data_engineering",
    "the-devops-handbook-how-to-create-world-class-agility-reliability-and-security-in-technology-organizations-978-1942788003_compress.pdf": "devops"
}

# List all PDF files
pdf_files = [f for f in os.listdir(content_dir) if f.lower().endswith('.pdf')]
if not pdf_files:
    logging.warning("No PDF files found in the 'docs' directory.")

# Extract and chunk text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Chunks of 500 characters
    chunk_overlap=100,  # 100-character overlap for better context
    separators=["\n\n", "\n", " ", ""],  # Preferred split points
)

all_documents = []
for pdf_file in pdf_files:
    file_path = os.path.join(content_dir, pdf_file)
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        continue
    
    # Get the topic for this PDF
    topic = topic_mapping.get(pdf_file, "unknown")
    
    logging.info(f"Processing file: {file_path} (Topic: {topic})")
    extracted_text = extract_text(file_path)
    
    if not extracted_text.strip():
        logging.warning(f"Skipping {pdf_file} - No text extracted.")
        continue
    
    chunks = text_splitter.split_text(extracted_text)
    if not chunks:
        logging.warning(f"Skipping {pdf_file} - No valid chunks created.")
        continue
    
    # Create Document objects with metadata
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={
                "source": pdf_file,
                "topic": topic
            }
        )
        all_documents.append(doc)

# Check if documents are available
if not all_documents:
    logging.error("No valid documents found for embedding.")
    exit()

# Load embedding model
logging.info("Loading embedding model...")
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

# Test embedding model
try:
    test_embedding = embeddings.embed_documents(["Test sentence"])
    if not test_embedding or len(test_embedding[0]) == 0:
        logging.error("Embedding model returned empty embeddings. Check Ollama model.")
        exit()
except Exception as e:
    logging.error(f"Embedding model failed: {e}")
    exit()

# Define vector DB directory
persist_directory = os.path.join(content_dir, "chroma")

# Create and persist the vector database
try:
    logging.info("Creating Chroma vector database with topic metadata...")
    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logging.info(f"Vector database created and saved successfully with {len(all_documents)} documents.")
except Exception as e:
    logging.error(f"Failed to create vector database: {e}")