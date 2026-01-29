from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from agent import process_query
import uvicorn
import time

app = FastAPI(
    title="UET AI Agent API",
    description="AI-powered assistant for UET Prospectus information",
    version="2.0"
)

# Add CORS middleware for web interface compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User's question")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    status: str = "success"

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "UET AI Agent API",
        "version": "2.0",
        "endpoints": {
            "/chat": "POST - Send questions about UET",
            "/health": "GET - Check service health"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for UET queries.
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        ChatResponse with AI-generated answer
    """
    try:
        start_time = time.time()
        
        # Process the query
        response_text = process_query(request.message)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            processing_time=round(processing_time, 2),
            status="success"
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    print("🚀 Starting UET AI Agent API...")
    print("📚 Make sure you've run ingest_improved.py first!")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")