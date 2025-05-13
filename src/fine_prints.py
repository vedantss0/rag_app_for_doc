from typing import List, Dict
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_TYPE, GOOGLE_API_KEY, OPENAI_API_KEY
from src.embeddings import EmbeddingManager
from src.document_loader import DocumentProcessor


class FinePrintsExtractor:
    """Extracts fine-prints (key details) from project documents"""
    
    def __init__(self):
        self.vector_store = EmbeddingManager().load_vector_store()
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if LLM_TYPE == "openai":
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        else:  # Default to Gemini
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    def extract_fine_prints(self) -> List[Dict]:
        """Extract fine-prints from all documents"""
        # Load and process all documents
        processor = DocumentProcessor()
        documents = processor.load_documents()
        
        # Define the extraction prompt
        extraction_prompt = PromptTemplate(
            input_variables=["document_content"],
            template="""
            You are an expert at analyzing project documents and extracting key details that are 
            critical for drafting project proposals. Please analyze the following document content 
            and extract the most important "fine-prints" - key details, requirements, 
            constraints, deadlines, budgets, or any other critical information.
            
            For each fine-print you extract, provide:
            1. A concise title/category
            2. The detailed information
            3. A brief explanation of why this is important for a project proposal
            
            Document content:
            {document_content}
            
            Please format your response as a list of fine-prints, with clear headers and details.
            """
        )
        
        # Create the extraction chain
        extraction_chain = LLMChain(llm=self.llm, prompt=extraction_prompt)
        
        all_fine_prints = []
        
        # Process each document to extract fine-prints
        for doc in documents:
            try:
                result = extraction_chain.run(document_content=doc.page_content)
                all_fine_prints.append({
                    "source": doc.metadata.get("source", "Unknown source"),
                    "page": doc.metadata.get("page", 0),
                    "fine_prints": result
                })
            except Exception as e:
                print(f"Error extracting fine-prints from document: {e}")
        
        return all_fine_prints
    
    def get_relevant_fine_prints(self, query: str, k: int = 5) -> List:
        """Get fine-prints relevant to a specific query"""
        # Search the vector store for relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Extract fine-prints from these relevant documents
        relevant_prompt = PromptTemplate(
            input_variables=["query", "document_content"],
            template="""
            You are an expert at analyzing project documents and extracting key details that are 
            critical for drafting project proposals. Based on the user query and the document content provided,
            extract the most important "fine-prints" - key details that are relevant to the query.
            
            User query: {query}
            
            Document content:
            {document_content}
            
            Please provide only the most relevant fine-prints to the query, formatted clearly.
            """
        )
        
        relevant_chain = LLMChain(llm=self.llm, prompt=relevant_prompt)
        
        results = []
        for doc in docs:
            try:
                result = relevant_chain.run(query=query, document_content=doc.page_content)
                results.append({
                    "source": doc.metadata.get("source", "Unknown source"),
                    "page": doc.metadata.get("page", 0),
                    "fine_prints": result
                })
            except Exception as e:
                print(f"Error extracting relevant fine-prints: {e}")
        
        return results