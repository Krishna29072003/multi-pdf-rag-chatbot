import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load env variables (GROQ_API_KEY)
load_dotenv()

st.set_page_config(page_title="Video Transcript Q&A", layout="wide")
st.title("🎥 Video Transcript Q&A (RAG)")

# ---------- Load text ----------
TEXT_PATH = Path("../data/text/video.txt")

@st.cache_data
def load_text():
    return TEXT_PATH.read_text(encoding="utf-8")

text = load_text()

# ---------- Split text ----------
@st.cache_data
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )
    return splitter.split_text(text)

chunks = split_text(text)

# ---------- Embeddings + Vector Store ----------
@st.cache_resource
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

vector_store = build_vector_store(chunks)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ---------- Prompt ----------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Using ONLY the transcript context below:
- Answer clearly and completely
- If the topic is discussed, summarize the key points in bullet points
- Do NOT add information that is not in the transcript

Transcript Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# ---------- UI ----------
question = st.text_input(
    "Ask a question about the video transcript:",
    placeholder="Was the topic of LLM discussed in the video?"
)

if st.button("Ask") and question:
    with st.spinner("Searching transcript and generating answer..."):
        relevant_chunks = retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in relevant_chunks)

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        answer = llm.invoke(final_prompt)

    st.subheader("📌 Answer")
    st.write(answer.content)

    with st.expander("🔍 Retrieved Context"):
        for i, doc in enumerate(relevant_chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content[:500])
