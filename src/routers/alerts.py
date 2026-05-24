from fastapi import APIRouter
from pydantic import BaseModel
import datetime

router = APIRouter()

class AlertPayload(BaseModel):
    user_email: str
    risk_level: str
    location: str
    distance_km: float

@router.post("/send")
async def send_evacuation_alert(payload: AlertPayload):
    # Phase 1: Simulated / Test Inbox
    # This logs the email instead of actually sending via SMTP right now.
    
    email_body = f"""
    URGENT FLOOD ALERT for {payload.location}
    Risk Level: {payload.risk_level.upper()}
    
    You are currently in a high-risk zone.
    Please evacuate immediately to the nearest safe shelter ({payload.distance_km} km away).
    
    Follow local NDMA guidelines.
    """
    
    print(f"--- MOCK EMAIL SENT TO {payload.user_email} ---")
    print(email_body)
    print("---------------------------------------")
    
    return {
        "success": True,
        "message": f"Simulated alert sent to {payload.user_email}",
        "timestamp": datetime.datetime.now().isoformat()
    }
