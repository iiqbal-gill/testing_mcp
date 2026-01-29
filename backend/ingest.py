import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
PDF_PATH = os.path.join(os.path.dirname(__file__), "../data/UET lahore Document.pdf")
DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")

# Department keywords for metadata tagging
DEPARTMENT_KEYWORDS = {
    "Computer Science": ["computer science", "cs department", "software", "programming", "algorithm", "data structure"],
    "Electrical Engineering": ["electrical engineering", "ee department", "electronics", "circuits", "power systems"],
    "Mechanical Engineering": ["mechanical engineering", "me department", "thermodynamics", "mechanics", "manufacturing"],
    "Civil Engineering": ["civil engineering", "ce department", "structures", "construction", "surveying"],
    "Chemical Engineering": ["chemical engineering", "chemistry", "process engineering", "chemical processes"],
    "Industrial Engineering": ["industrial engineering", "ie department", "operations", "production"],
    "Architecture": ["architecture", "architectural", "design studio", "building design"],
    "Petroleum Engineering": ["petroleum", "oil", "gas", "reservoir"],
    "Biomedical Engineering": ["biomedical", "medical devices", "biomechanics"],
    "Environmental Engineering": ["environmental", "water treatment", "pollution"],
    "Textile Engineering": ["textile", "fabric", "fiber"],
    "Metallurgy": ["metallurgy", "materials science", "metals"],
    "Mining Engineering": ["mining", "mineral extraction"],
    "Aerospace Engineering": ["aerospace", "aeronautics", "aircraft"],
}

def extract_department_metadata(text: str) -> list:
    """Extract department tags from text content."""
    departments = []
    text_lower = text.lower()
    
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            departments.append(dept)
    
    return departments if departments else ["General"]

def extract_section_type(text: str) -> str:
    """Identify what type of information this chunk contains."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["admission", "eligibility", "requirement", "criteria"]):
        return "admissions"
    elif any(word in text_lower for word in ["fee", "tuition", "cost", "charges"]):
        return "fees"
    elif any(word in text_lower for word in ["course", "syllabus", "curriculum", "semester"]):
        return "curriculum"
    elif any(word in text_lower for word in ["faculty", "professor", "dr.", "lecturer", "head of department"]):
        return "faculty"
    elif any(word in text_lower for word in ["scholarship", "financial aid", "merit"]):
        return "financial_aid"
    elif any(word in text_lower for word in ["hostel", "accommodation", "residence"]):
        return "facilities"
    else:
        return "general_info"

def clean_text(text: str) -> str:
    """Clean and normalize text from PDF."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*\n', '', text)
    # Fix common OCR issues
    text = text.replace('ï¬', 'fi').replace('ï¬‚', 'fl')
    return text.strip()

def ingest_data():
    print("🚀 Starting Enhanced Data Ingestion...")
    
    # 1. Load PDF
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Please add the file.")
    
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages.")

    # 2. Clean documents
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    
    # 3. Split Text with BETTER parameters
    # Smaller chunks with more overlap for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for more precise retrieval
        chunk_overlap=300,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""],  # Better splitting points
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(splits)} chunks.")

    # 4. Add metadata to each chunk
    print("🏷️ Adding metadata to chunks...")
    for i, split in enumerate(splits):
        # Add department tags
        departments = extract_department_metadata(split.page_content)
        split.metadata["departments"] = ",".join(departments)
        
        # Add section type
        split.metadata["section_type"] = extract_section_type(split.page_content)
        
        # Add chunk ID for tracking
        split.metadata["chunk_id"] = i
        
        # Keep original page number
        if "page" not in split.metadata:
            split.metadata["page"] = "unknown"
    
    print(f"📊 Metadata added. Sample: {splits[0].metadata if splits else 'None'}")

    # 5. Create Vector Store with improved embeddings
    print("⏳ Generating Embeddings (this may take a moment)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if os.path.exists(DB_PATH):
        print("🗑️ Clearing old database...")
        import shutil
        shutil.rmtree(DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"}  # Better for semantic search
    )
    
    print(f"💾 Vector Database saved to {DB_PATH}")
    print(f"✅ Ingestion complete! Total chunks: {len(splits)}")
    
    # Print department distribution
    dept_counts = {}
    for split in splits:
        for dept in split.metadata["departments"].split(","):
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    print("\n📊 Department Distribution:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {dept}: {count} chunks")

if __name__ == "__main__":
    ingest_data()