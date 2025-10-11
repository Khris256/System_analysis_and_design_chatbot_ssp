import streamlit as st
import pickle
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# IMPORTANT: API key is stored in Streamlit Cloud Secrets
# Go to: App Settings > Secrets > Add GOOGLE_API_KEY
# This code will NOT expose your API key in the public repo
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets!")
    st.info("Please add your API key in Streamlit Cloud: Settings > Secrets")
    st.stop()

st.set_page_config(
    page_title="System Analysis Chatbot",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        color: #FAFAFA;
        background-color:#020203;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stChatMessage {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #0056b3;
        text-align: right;
    }
    .stChatMessage.assistant {
        background-color: #262730;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("System Analysis Chatbot")

VECTOR_STORE_PATH = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pkl"

@st.cache_resource
def load_embeddings():
    """
    Load the EXACT same embedding model used during vector store creation.
    Model: sentence-transformers/all-MiniLM-L6-v2
    Device: CPU (as specified in main.py)
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs
    )
    return embeddings

@st.cache_resource
def load_vector_store(path):
    """Load the pre-created vector store from pickle file"""
    if not os.path.exists(path):
        st.error(f"❌ Vector store file not found: {path}")
        st.error(f"Current directory: {os.getcwd()}")
        st.error(f"Available files: {os.listdir('.')}")
        st.stop()
    
    file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
    
    try:
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        st.success(f"✅ Vector store loaded successfully! ({file_size:.2f} MB)")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        st.stop()

# Load resources
embeddings = load_embeddings()
vector_store = load_vector_store(VECTOR_STORE_PATH)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about System Analysis and Design:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Use retriever for better FAISS compatibility
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            docs = retriever.get_relevant_documents(prompt)
            
            if not docs:
                full_response = "I couldn't find relevant information in the document. Please try rephrasing your question."
            else:
                # Create LLM and QA chain
                llm = ChatGoogleGenerativeAI(
                    temperature=0, 
                    model="gemini-2.0-flash-exp"
                )
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                
                # Get response from the chain
                response = chain.run(input_documents=docs, question=prompt)
                full_response = response
            
        except Exception as e:
            full_response = f"❌ An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
            st.error(f"Error type: {type(e).__name__}")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})