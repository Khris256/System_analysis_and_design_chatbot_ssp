"""
Simple script to create vector store with FAISS compatibility
This only creates the .pkl file - no LLM queries needed
"""

from PyPDF2 import PdfReader
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    
    # Create embeddings - SAME AS MAIN.PY
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    
    # Create FAISS vector store
    print("Creating FAISS vector store (this may take a few minutes)...")
    vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Test the vector store works before saving
    print("Testing vector store...")
    try:
        test_results = vectorStore.similarity_search("test query", k=2)
        print(f"✅ Vector store test successful! Found {len(test_results)} results")
    except Exception as e:
        print(f"⚠️ Warning: Vector store test failed: {e}")
        print("Continuing anyway...")
    
    # Save to disk
    output_file = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pkl"
    print(f"Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(vectorStore, f)
    
    # Verify saved file
    print(f"Verifying saved file...")
    with open(output_file, "rb") as f:
        loaded_store = pickle.load(f)
    print(f"✅ File verified successfully!")
    
    print(f"\n✅ SUCCESS! Vector store saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Print FAISS version info
    try:
        import faiss
        print(f"FAISS version: {faiss.__version__}")
    except:
        print("FAISS version: Unable to determine")

if __name__ == "__main__":
    import os
    import sys
    
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use default path (update this to your PDF location)
        pdf_path = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ ERROR: PDF file not found at: {pdf_path}")
        print("\nUsage: python create_vector_store.py <path_to_pdf>")
        print("\nOr place your PDF in the same folder and name it:")
        print("Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pdf")
        sys.exit(1)
    
    create_vector_store(pdf_path)