import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

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

st.title("SSP.ai 💡")

# Load the pre-vectorized PDF data
VECTOR_STORE_PATH = "Systems_Analysis_and_Design_Ninth_Edition_Gary_B_Shelly_Harry_J_Rosenblatt.pkl"

@st.cache_resource
def load_vector_store(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        st.success("Notes  loaded successfully! , Ask your questions below.")
        return vector_store
    else:
        st.error(f"Vector store not found at {path}.")
        st.stop()

vector_store = load_vector_store(VECTOR_STORE_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about System Analysis and Design:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if vector_store: # Check if vector_store was loaded successfully
            docs = vector_store.similarity_search(query=prompt)
            llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=prompt)
            full_response = response
        else:
            full_response = "Error: Vector store not loaded."
        
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
