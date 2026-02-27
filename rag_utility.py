import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import unstructured, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import  ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    model= "llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def process_document_to_chroma_db(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    texts = text_splitter.split_documents(documents)

    # If vector DB doesn't exist → create it
    if "vectorDB" not in st.session_state:
        st.session_state.vectorDB = Chroma.from_documents(
            documents=texts,
            embedding=embedding
        )
    else:
        # If it already exists → add documents
        st.session_state.vectorDB.add_documents(texts)


def answer_question(user_question):
    vectordb = st.session_state.vectorDB
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type= "stuff",
       retriever=retriever,
       return_source_documents=True
    )
    response = qa_chain.invoke({"query":user_question})
    answer = response["result"]
    sources = response.get("source_documents", [])  # list of Document objects

    return answer, sources
