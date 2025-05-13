import os
from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import VECTOR_STORE_PATH, OPENAI_API_KEY, LLM_TYPE
from src.document_loader import DocumentProcessor


class EmbeddingManager:
    """Manages document embeddings and vector store operations"""
    
    def __init__(self):
        if LLM_TYPE == "openai":
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.embeddings = OpenAIEmbeddings()
        else:
            # Default to a free local model if not using OpenAI
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def create_vector_store(self, documents: List) -> FAISS:
        """Create a vector store from documents"""
        return FAISS.from_documents(documents, self.embeddings)
    
    def save_vector_store(self, vector_store: FAISS) -> None:
        """Save the vector store to disk"""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
    
    def load_vector_store(self) -> FAISS:
        """Load the vector store from disk"""
        if not os.path.exists(VECTOR_STORE_PATH):
            # If vector store doesn't exist, create and save it
            processor = DocumentProcessor()
            documents = processor.load_documents()
            chunks = processor.split_documents(documents)
            vector_store = self.create_vector_store(chunks)
            self.save_vector_store(vector_store)
            return vector_store
        
        return FAISS.load_local(VECTOR_STORE_PATH, self.embeddings)