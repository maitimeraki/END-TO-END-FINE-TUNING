
from pydantic import BaseModel, Field

# Define the request body schema
class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
