import os
import glob
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Class to load and process documents from a directory"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the data directory"""
        pdf_pattern = os.path.join(self.data_dir, "*.pdf")
        return glob.glob(pdf_pattern)
    
    def load_documents(self) -> List:
        """Load all documents from the data directory"""
        pdf_files = self.get_pdf_files()
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}")
        
        documents = []
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)