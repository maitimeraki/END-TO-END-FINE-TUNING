import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from threading import Thread
from src.service.inference import engine  # Your logic moved to src/inference.py
from config import ChatRequest  # Your Pydantic model for request validation

app = FastAPI(title="LLM Production API")

# 1. Security: Simple API Key check
# API_KEY = os.getenv("HF_TOKEN", "default-secret-key")
# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# async def get_api_key(header_key: str = Security(api_key_header)):
#     if header_key == API_KEY:
#         return header_key
#     raise HTTPException(status_code=403, detail="Could not validate credentials")

# 2. Monitoring: Prometheus Metrics
Instrumentator().instrument(app).expose(app)

@app.post("/v1/chat")
async def chat(
    request: ChatRequest, 
    # api_key: str = Depends(get_api_key)
):
    """
    Main Chat Endpoint with Streaming
    """
    try:
        inputs_text = engine.chat_template_format(request.prompt)
        # We pass the generator function to StreamingResponse
        return StreamingResponse(
            engine.generate_stream(inputs_text),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/stop")
async def stop_chat_generation():
    """
    Stop the current in-progress generation request.
    """
    engine.request_stop()
    return {"status": "stopping"}
    
    
    
if __name__ == "__main__":
    import uvicorn
    # In production, use 1 worker for GPU stability
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)