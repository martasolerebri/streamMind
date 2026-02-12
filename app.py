import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="MathMind Pro", page_icon="🧮", layout="wide")
st.title("🧮 MathMind Pro: RAG Math Tutor")

with st.sidebar:
    st.header("Credentials")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    st.divider()
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Math PDF", type="pdf")
    if st.button("Clear Database"):
        for key in ["math_retriever", "messages"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

if not groq_api_key or not hf_api_key:
    st.warning("Please enter your API Keys in the sidebar.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key):
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key)

def process_pdf(file):
    # Creamos un archivo temporal físico para que el loader pueda leerlo por ruta
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file.getbuffer())
        temp_path = tf.name
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        return vs.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error indexing PDF: {e}")
        return None
    finally:
        # Limpiamos el archivo temporal después de usarlo
        if os.path.exists(temp_path):
            os.remove(temp_path)

if uploaded_file and "math_retriever" not in st.session_state:
    with st.spinner("Indexing math content..."):
        retriever = process_pdf(uploaded_file)
        if retriever:
            st.session_state.math_retriever = retriever
            st.success("Knowledge base updated!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about your math PDF"):
    if "math_retriever" not in st.session_state:
        st.error("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        template = """You are an expert Math Tutor. Use the provided context to answer.
        Always use LaTeX for math ($ inline, $$ block).
        Context: {context}
        Question: {input}
        Answer:"""
        
        qa_prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": st.session_state.math_retriever, "input": RunnablePassthrough()}
            | qa_prompt | llm | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})