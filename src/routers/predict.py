from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from src.services.ml import predict_flood, get_calibrated_risk, fetch_weather, analyze_drivers
from src.services.irsa_scraper import scrape_irsa_daily

router = APIRouter()

class LocationPayload(BaseModel):
    lat: float
    lng: float

class FeaturesPayload(BaseModel):
    features: Dict[str, int]

@router.post("/location")
async def predict_by_location(payload: LocationPayload):
    weather = fetch_weather(payload.lat, payload.lng)
    irsa_data = scrape_irsa_daily()
    
    # Base mapping
    monsoon_intensity = min(10, int(weather.get('rain_1h', 0) / 2) + 4)
    river_management = 5
    dams_quality = 5
    
    # If we have scraped IRSA data, adjust river/dam features based on total inflows
    if irsa_data and irsa_data.get("success"):
        total_inflow = irsa_data.get("rim_station_inflows", 150000)
        # Mock logic: if inflows are extremely high, river management stress increases
        if total_inflow > 250000:
            river_management = 2  # Poor/stressed
            dams_quality = 3      # Stressed capacity
        elif total_inflow > 180000:
            river_management = 4
            dams_quality = 5
        else:
            river_management = 7  # Good handling
            dams_quality = 7
    
    # Map weather and IRSA to XGBoost features
    features = {
        'MonsoonIntensity': monsoon_intensity,
        'RiverManagement': river_management,
        'DamsQuality': dams_quality,
        'TopographyDrainage': 5,
        'Urbanization': 6,
        'ClimateChange': 7,
    }
    
    prob = predict_flood(features)
    risk_level, confidence = get_calibrated_risk(prob)
    drivers = analyze_drivers(features)
    
    return {
        "success": True,
        "flood_probability": prob,
        "risk_level": risk_level,
        "confidence": confidence,
        "drivers": drivers,
        "weather": weather,
        "irsa_synced": True if irsa_data else False
    }

@router.post("/manual")
async def predict_manual(payload: FeaturesPayload):
    prob = predict_flood(payload.features)
    risk_level, confidence = get_calibrated_risk(prob)
    drivers = analyze_drivers(payload.features)
    
    return {
        "success": True,
        "flood_probability": prob,
        "risk_level": risk_level,
        "confidence": confidence,
        "drivers": drivers
    }
