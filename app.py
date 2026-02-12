import streamlit as st
import tempfile
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="StreamMind AI", page_icon="🎬", layout="wide")
st.title("🎬 StreamMind: Chat with your Videos")
st.markdown("Analyze YouTube transcripts instantly using **Groq** (Llama 3) and **RAG**.")

with st.sidebar:
    st.header("Credentials")
    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_api_key = st.text_input("Hugging Face API Key", type="password")
    
    st.divider()
    
    st.header("Content Source")
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    st.divider()
    st.info("This system extracts the transcript, indexes it into a vector store, and allows you to perform lightning-fast queries.")

if not groq_api_key or not hf_api_key:
    st.warning("Please enter your API Keys in the sidebar to begin.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key):
    llm = ChatGroq(
        api_key=groq_key, 
        model="llama-3.3-70b-versatile", 
        temperature=0.3
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key)

def process_video(url):
    try:
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=True,
            language=["en", "es"]
        )
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

if video_url:
    if "current_video" not in st.session_state or st.session_state.current_video != video_url:
        with st.spinner("Transcribing and indexing video..."):
            retriever = process_video(video_url)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.current_video = video_url
                st.session_state.messages = [] 
                st.success("Video ready for questions!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("What would you like to know about the video?"):
    if "retriever" not in st.session_state:
        st.error("Please provide a valid YouTube URL first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        system_prompt = (
            "You are an expert analytical assistant. Answer questions based exclusively on the "
            "provided video transcript. If the information is not in the context, "
            "state that the video does not mention it. Keep answers clear and structured.\n\n"
            "Transcript Context:\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = (
            {"context": st.session_state.retriever, "input": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing key moments..."):
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})