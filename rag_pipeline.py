import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
#from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
#from langchain_community.retrievers import MultiQueryRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

#PDF_DIR = Path("docs")
#PDF_DIR = Path(__file__).resolve().parent.parent / "docs"
PDF_DIR = Path(__file__).resolve().parent.parent / "docs"


documents = []

for pdf_file in PDF_DIR.rglob("*.pdf"):
    print(f"Loading: {pdf_file.name}")
    
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()   # each page = Document object
    
    documents.extend(pages)

print(f"\nTotal pages loaded: {len(documents)}")


'''from collections import Counter

sources = [doc.metadata["source"] for doc in documents]
print(Counter(sources))'''




splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=90

)

chunks = splitter.split_documents(documents)
#print(len(chunks))

'''for i in chunks:
    print(i.page_content[:100])'''


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#retriever.invoke("What is the main topic of the documents?")
#for i in a:
    #print(i.page_content[:100])

#print(retriever.invoke("What is the main topic of the documents?"))

prompt = PromptTemplate(
    template="""

You are a strict retrieval-based assistant.
Rules you MUST follow:
1. Use ONLY the provided context.
2. Answer ONLY what is directly relevant to the question.
3. If multiple subtopics appear, include ONLY those that clearly answer the question.
4. Ignore dataset references, citations, or tangential mentions unless they are central.
5. If the context is too broad, say so explicitly.

Context:
{context}

Question:
{question}

Answer (bullet points only):
""",
    input_variables=["context", "question"]
)

question = "Explain about LLMs?"

relevant_chunks = retriever.invoke(question)

formatted_prompt = prompt.format(context="\n\n".join([chunk.page_content for chunk in relevant_chunks]), question=question)

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2
)


response = llm.invoke(formatted_prompt)
print("\nResponse:\n", response.content)