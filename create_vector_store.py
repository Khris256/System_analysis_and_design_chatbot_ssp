"""
Create vector store using ChromaDB (no NumPy compatibility issues)
"""

from PyPDF2 import PdfReader
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vector_store(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    
    # Read PDF
    pdf_reader = PdfReader(pdf_path)
    
    # Extract text
    text = ""
    for i, page in enumerate(pdf_reader.pages):
        print(f"Processing page {i+1}/{len(pdf_reader.pages)}...")
        text += page.extract_text()
    
    print(f"Total text extracted: {len(text)} characters")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    print(f"Created {len(chunks)} text chunks")
    
    # Create embeddings
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    
    # Create Chroma vector store (no FAISS issues!)
    print("Creating Chroma vector store...")
    persist_directory = "./chroma_db"
    vectorStore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Test the vector store
    print("Testing vector store...")
    try:
        test_results = vectorStore.similarity_search("test query", k=2)
        print(f"✅ Vector store test successful! Found {len(test_results)} results")
    except Exception as e:
        print(f"⚠️ Warning: Vector store test failed: {e}")
    
    # Also save as pickle for compatibility
    output_file = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pkl"
    print(f"Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(vectorStore, f)
    
    print(f"\n✅ SUCCESS!")
    print(f"- ChromaDB saved to: {persist_directory}")
    print(f"- Pickle file: {output_file} ({os.path.getsize(output_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at: {pdf_path}")
        print("\nUsage: python create_vector_store_chromadb.py <path_to_pdf>")
        sys.exit(1)
    
    create_vector_store(pdf_path)