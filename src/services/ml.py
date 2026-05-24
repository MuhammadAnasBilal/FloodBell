import joblib
import pandas as pd
import os
import requests

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/xgb_production.pkl')
OWM_KEY = 'd1ccaf30c2e39ba82506f4ed9bdb50f2'

model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Loaded {MODEL_PATH}")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")

# Base features for regions (simplified fallback if no API)
REGION_BASELINES = {
    'karachi': {'TopographyDrainage': 2, 'Urbanization': 8, 'CoastalVulnerability': 7, 'DrainageSystems': 3},
    'nowshera': {'TopographyDrainage': 4, 'RiverManagement': 3, 'DamsQuality': 4},
    'sukkur': {'RiverManagement': 4, 'DamsQuality': 3, 'AgriculturalPractices': 6},
    'default': {'TopographyDrainage': 5, 'RiverManagement': 5, 'Urbanization': 5}
}

def fetch_weather(lat: float, lng: float) -> dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&units=metric&appid={OWM_KEY}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'rain_1h': data.get('rain', {}).get('1h', 0)
            }
    except Exception as e:
        print(f"Weather fetch failed: {e}")
    return {'temp': 30, 'humidity': 50, 'wind_speed': 2, 'rain_1h': 0}

def predict_flood(features: dict) -> float:
    if not model:
        return 0.5
    feature_order = [
        'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
        'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
        'Siltation', 'AgriculturalPractices', 'Encroachments',
        'IneffectiveDisasterPreparedness', 'DrainageSystems',
        'CoastalVulnerability', 'Landslides', 'Watersheds',
        'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
        'InadequatePlanning', 'PoliticalFactors'
    ]
    df = pd.DataFrame([{f: features.get(f, 5) for f in feature_order}])
    prob = model.predict(df)[0]
    return float(prob)

def get_calibrated_risk(probability: float) -> tuple[str, int]:
    """Returns risk level and confidence percentage."""
    # Based on the user's calibration requirements
    confidence = int((abs(probability - 0.5) * 2) * 100)
    # Clamp confidence
    confidence = max(50, min(95, confidence + 50)) 
    
    if probability > 0.625:
        return "critical", confidence
    elif probability >= 0.57:
        return "high", confidence
    elif probability >= 0.54:
        return "medium", confidence
    else:
        return "low", confidence

def analyze_drivers(features: dict) -> list[str]:
    """Determine what's driving the risk up or down."""
    drivers = []
    if features.get('MonsoonIntensity', 5) >= 7: drivers.append("↑ High Rainfall")
    if features.get('DrainageSystems', 5) <= 3: drivers.append("↓ Poor Drainage")
    if features.get('RiverManagement', 5) <= 4: drivers.append("↓ River Levels")
    if not drivers: drivers.append("Stable Conditions")
    return drivers
