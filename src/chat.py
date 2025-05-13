import os
from typing import Dict, Any, List

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

from src.config import LLM_TYPE, GOOGLE_API_KEY, OPENAI_API_KEY
from src.embeddings import EmbeddingManager


class ChatInterface:
    """Chat interface for interacting with project documents"""
    
    def __init__(self):
        self.vector_store = EmbeddingManager().load_vector_store()
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = self._create_qa_chain()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if LLM_TYPE == "openai":
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            return ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        else:  # Default to Gemini
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    def _create_qa_chain(self):
        """Create a QA chain for document retrieval and answering"""
        # Create a custom prompt template for project proposal generation
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful project proposal assistant. Use the following context from project documents
            to answer the user's question. If you don't know the answer based on the provided context,
            just say that you don't have enough information, but try to be helpful and suggest what
            might be relevant to include in a project proposal.
            
            Context:
            {context}
            
            User question: {question}
            
            Your detailed answer:
            """
        )
        
        # Create a document QA chain with the custom prompt
        doc_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=qa_prompt
        )
        
        # Create the conversational retrieval chain
        retrieval_chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            combine_docs_chain=doc_chain,
            memory=self.memory
        )
        
        return retrieval_chain
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the response"""
        try:
            result = self.qa_chain({"question": query})
            return {
                "query": query,
                "response": result["answer"],
                "source_documents": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", 0),
                        "content": doc.page_content[:200] + "..."  # First 200 chars for reference
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "query": query,
                "response": f"Error processing your query: {str(e)}",
                "source_documents": []
            }