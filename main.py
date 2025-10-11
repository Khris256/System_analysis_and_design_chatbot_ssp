import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#sidebar contents
with st.sidebar:
    st.title("SSP assistant😊")
    st.markdown('''
        ##About
        This app was designed by ssp students to easen their revision process 
    ''')
    add_vertical_space()
    st.write('Made by khris calvin')

def main():
    load_dotenv()
    st.header("ssp.ai")

    #Upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
   
    

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
            #st.write("Embeddings Loaded from the disk")
        else:
            #embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2" #all-MiniLM-L6-v2 -all-MiniLM-L12-v2   all-mpnet-base-v2
            model_kwargs = {'device':'cpu'}
            embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                 pickle.dump(vectorStore, f)
            #st.write("Embeddings saved to the disk")

            #Take user input
        query = st.text_input("Ask questions about system analysis and design") 
        #st.write(query)

        if query:
            docs = vectorStore.similarity_search(query=query,)
            st.write(docs)
            llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
           

        
     


if __name__ == "__main__":
    main()


      