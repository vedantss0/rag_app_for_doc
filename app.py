import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from src.fine_prints_graph import FinePrintsGraph
from src.chat_graph import ChatGraph
from src.utils import ensure_directories, save_to_file, format_fine_prints_for_output

# Create app instance
app = FastAPI(
    title="Project Proposal Assistant API",
    description="API for extracting fine-prints from project documents and answering queries for project proposals",
    version="1.0.0"
)

# Initialize directories
ensure_directories()

# Define request/response models
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    response: str
    source_documents: Optional[List[Dict]] = []


class FinePrintsResponse(BaseModel):
    fine_prints: List[Dict]


# Create API endpoints
@app.get("/")
async def root():
    return {"message": "Project Proposal Assistant API is running"}


@app.get("/fine-prints", response_model=FinePrintsResponse)
async def get_fine_prints():
    """Extract and return fine-prints from all project documents"""
    try:
        extractor = FinePrintsGraph()
        fine_prints = extractor.extract()
        
        # Save fine prints to a file
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_to_file(
            format_fine_prints_for_output(fine_prints), 
            os.path.join(output_dir, "fine_prints.txt")
        )
        
        return {"fine_prints": fine_prints}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting fine-prints: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user query and return response"""
    try:
        chat_interface = ChatGraph()
        response = chat_interface.process_query(request.query)
        
        # Append the query and response to chat log file
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(output_dir, "chat_response.txt"), "a", encoding="utf-8") as f:
            f.write(f"Query: {request.query}\n")
            f.write(f"Response: {response['response']}\n")
            f.write("-" * 80 + "\n\n")
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)