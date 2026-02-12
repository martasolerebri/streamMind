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
    st.warning("Please enter your API Keys in the sidebar to start.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key):
    # Model optimized for math reasoning
    llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key)

def process_pdf_safe(file):
    # We create a physical temporary file to ensure compatibility
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file.getbuffer())
        temp_path = tf.name
    
    try:
        # Using PyPDFLoader specifically as it's the most stable for Streamlit Cloud
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        return vs.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Critical error loading PDF: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Logic to prevent re-processing
if uploaded_file and "math_retriever" not in st.session_state:
    with st.spinner("Indexing mathematical formulas..."):
        retriever = process_pdf_safe(uploaded_file)
        if retriever:
            st.session_state.math_retriever = retriever
            st.success("PDF knowledge integrated!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a math question about your PDF"):
    if "math_retriever" not in st.session_state:
        st.error("Upload a PDF file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Prompt focusing on LaTeX rendering
        template = """You are a Math expert. Use the following context to answer.
        ALWAYS use LaTeX for formulas (wrap with $$ for blocks, $ for inline).
        
        Context: {context}
        Question: {input}
        Detailed Solution:"""
        
        qa_prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": st.session_state.math_retriever, "input": RunnablePassthrough()}
            | qa_prompt | llm | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Reasoning..."):
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})