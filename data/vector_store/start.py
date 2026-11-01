import time
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from config.settings import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

settings = get_settings()

@st.cache_resource(show_spinner=False, ttl=3600)
def get_embeddings():
    """Returns a singleton instance of HuggingFaceEmbeddings."""
    
    _embeddings = HuggingFaceEmbeddings(model=settings.embedding_model)

    return _embeddings

def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = 384
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = [x["name"] for x in pc.list_indexes()]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)

    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)

def vectorstore():
    """Returns a singleton instance of PineconeVectorStore."""
    index=ensure_index()
    _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore

def get_retriever():
    return vectorstore().as_retriever(search_kwargs={"k": settings.top_k})

def add_document(chunks):
    store=vectorstore()
    return store.add_documents(chunks)