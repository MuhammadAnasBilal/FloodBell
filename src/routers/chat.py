from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ChatPayload(BaseModel):
    message: str
    lat: float | None = None
    lng: float | None = None

@router.post("/")
async def chat_endpoint(payload: ChatPayload):
    from src.providers.ai_assistant import generate_chat_response
    
    # In V3, we route the chat intent to different sub-prompts.
    # For now, it delegates to the existing Grok integration.
    # In a full implementation, this uses the router pattern.
    response = generate_chat_response(payload.message, {"location": f"{payload.lat}, {payload.lng}"})
    
    return {
        "success": True,
        "response": response
    }
