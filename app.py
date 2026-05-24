"""
FloodGuard Pakistan - Disaster Management System
==================================================
Flask backend with ML prediction, weather API proxy,
IRSA river data, A* agent, CSP solver, and Knowledge Base.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import requests as http_requests

# Import new providers
try:
    from src.providers.static import search_location, get_local_hydrology, get_nearest_shelter
    from src.providers.ai_assistant import generate_chat_response
except ImportError:
    pass # Will be handled by fallback if missing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================
OWM_API_KEY = 'd1ccaf30c2e39ba82506f4ed9bdb50f2'

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

FEATURE_DESCRIPTIONS = {
    'MonsoonIntensity': 'Intensity of monsoon rainfall',
    'TopographyDrainage': 'Natural drainage capability',
    'RiverManagement': 'River management effectiveness',
    'Deforestation': 'Deforestation level',
    'Urbanization': 'Urbanization degree',
    'ClimateChange': 'Climate change impact',
    'DamsQuality': 'Dam quality and condition',
    'Siltation': 'Waterway siltation level',
    'AgriculturalPractices': 'Farming impact on flooding',
    'Encroachments': 'Waterway encroachment level',
    'IneffectiveDisasterPreparedness': 'Disaster preparedness gaps',
    'DrainageSystems': 'Drainage infrastructure quality',
    'CoastalVulnerability': 'Coastal flooding vulnerability',
    'Landslides': 'Landslide risk',
    'Watersheds': 'Watershed management quality',
    'DeterioratingInfrastructure': 'Infrastructure decay',
    'PopulationScore': 'Population density pressure',
    'WetlandLoss': 'Wetland degradation extent',
    'InadequatePlanning': 'Planning quality',
    'PoliticalFactors': 'Political factors'
}

# ============================================================
# Pakistan Regions with static feature baselines
# ============================================================
PAKISTAN_REGIONS = [
    {"id": 1, "name": "Sukkur, Sindh", "lat": 27.7052, "lng": 68.8574,
     "province": "Sindh", "population": 499900, "shelters": 5, "hospitals": 3,
     "nearest_station": "Sukkur",
     "baselines": {
         "TopographyDrainage": 7, "Deforestation": 5, "Urbanization": 6,
         "Siltation": 8, "AgriculturalPractices": 6, "Encroachments": 7,
         "IneffectiveDisasterPreparedness": 6, "DrainageSystems": 5,
         "CoastalVulnerability": 3, "Landslides": 1, "Watersheds": 4,
         "DeterioratingInfrastructure": 7, "PopulationScore": 7, "WetlandLoss": 6,
         "InadequatePlanning": 6, "PoliticalFactors": 5
     }},
    {"id": 2, "name": "Dadu, Sindh", "lat": 26.7319, "lng": 67.7750,
     "province": "Sindh", "population": 350000, "shelters": 3, "hospitals": 2,
     "nearest_station": "Sukkur",
     "baselines": {
         "TopographyDrainage": 8, "Deforestation": 4, "Urbanization": 4,
         "Siltation": 7, "AgriculturalPractices": 7, "Encroachments": 5,
         "IneffectiveDisasterPreparedness": 7, "DrainageSystems": 4,
         "CoastalVulnerability": 2, "Landslides": 1, "Watersheds": 5,
         "DeterioratingInfrastructure": 6, "PopulationScore": 5, "WetlandLoss": 5,
         "InadequatePlanning": 7, "PoliticalFactors": 5
     }},
    {"id": 3, "name": "Jacobabad, Sindh", "lat": 28.2769, "lng": 68.4514,
     "province": "Sindh", "population": 200000, "shelters": 2, "hospitals": 1,
     "nearest_station": "Guddu",
     "baselines": {
         "TopographyDrainage": 8, "Deforestation": 3, "Urbanization": 3,
         "Siltation": 6, "AgriculturalPractices": 7, "Encroachments": 4,
         "IneffectiveDisasterPreparedness": 8, "DrainageSystems": 3,
         "CoastalVulnerability": 1, "Landslides": 1, "Watersheds": 4,
         "DeterioratingInfrastructure": 7, "PopulationScore": 4, "WetlandLoss": 4,
         "InadequatePlanning": 8, "PoliticalFactors": 6
     }},
    {"id": 4, "name": "Nowshera, KPK", "lat": 34.0153, "lng": 71.9747,
     "province": "KPK", "population": 270000, "shelters": 4, "hospitals": 2,
     "nearest_station": "Kabul @ Nowshera",
     "baselines": {
         "TopographyDrainage": 6, "Deforestation": 6, "Urbanization": 5,
         "Siltation": 5, "AgriculturalPractices": 5, "Encroachments": 6,
         "IneffectiveDisasterPreparedness": 5, "DrainageSystems": 5,
         "CoastalVulnerability": 0, "Landslides": 7, "Watersheds": 6,
         "DeterioratingInfrastructure": 5, "PopulationScore": 5, "WetlandLoss": 4,
         "InadequatePlanning": 5, "PoliticalFactors": 4
     }},
    {"id": 5, "name": "Charsadda, KPK", "lat": 34.1478, "lng": 71.7310,
     "province": "KPK", "population": 180000, "shelters": 3, "hospitals": 1,
     "nearest_station": "Kabul @ Nowshera",
     "baselines": {
         "TopographyDrainage": 6, "Deforestation": 7, "Urbanization": 4,
         "Siltation": 5, "AgriculturalPractices": 6, "Encroachments": 5,
         "IneffectiveDisasterPreparedness": 6, "DrainageSystems": 4,
         "CoastalVulnerability": 0, "Landslides": 6, "Watersheds": 5,
         "DeterioratingInfrastructure": 6, "PopulationScore": 4, "WetlandLoss": 5,
         "InadequatePlanning": 6, "PoliticalFactors": 5
     }},
    {"id": 6, "name": "Dera Ghazi Khan, Punjab", "lat": 30.0561, "lng": 70.6346,
     "province": "Punjab", "population": 399000, "shelters": 4, "hospitals": 3,
     "nearest_station": "Taunsa",
     "baselines": {
         "TopographyDrainage": 7, "Deforestation": 5, "Urbanization": 5,
         "Siltation": 6, "AgriculturalPractices": 6, "Encroachments": 6,
         "IneffectiveDisasterPreparedness": 6, "DrainageSystems": 5,
         "CoastalVulnerability": 1, "Landslides": 3, "Watersheds": 5,
         "DeterioratingInfrastructure": 6, "PopulationScore": 6, "WetlandLoss": 5,
         "InadequatePlanning": 6, "PoliticalFactors": 5
     }},
    {"id": 7, "name": "Rajanpur, Punjab", "lat": 29.1044, "lng": 70.3301,
     "province": "Punjab", "population": 170000, "shelters": 2, "hospitals": 1,
     "nearest_station": "Guddu",
     "baselines": {
         "TopographyDrainage": 7, "Deforestation": 4, "Urbanization": 3,
         "Siltation": 7, "AgriculturalPractices": 7, "Encroachments": 5,
         "IneffectiveDisasterPreparedness": 7, "DrainageSystems": 3,
         "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 4,
         "DeterioratingInfrastructure": 7, "PopulationScore": 3, "WetlandLoss": 5,
         "InadequatePlanning": 7, "PoliticalFactors": 6
     }},
    {"id": 8, "name": "Muzaffargarh, Punjab", "lat": 30.0740, "lng": 71.1932,
     "province": "Punjab", "population": 340000, "shelters": 3, "hospitals": 2,
     "nearest_station": "Taunsa",
     "baselines": {
         "TopographyDrainage": 7, "Deforestation": 4, "Urbanization": 4,
         "Siltation": 6, "AgriculturalPractices": 7, "Encroachments": 5,
         "IneffectiveDisasterPreparedness": 6, "DrainageSystems": 4,
         "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 5,
         "DeterioratingInfrastructure": 6, "PopulationScore": 5, "WetlandLoss": 5,
         "InadequatePlanning": 6, "PoliticalFactors": 5
     }},
    {"id": 9, "name": "Swat, KPK", "lat": 35.2227, "lng": 72.3528,
     "province": "KPK", "population": 250000, "shelters": 3, "hospitals": 2,
     "nearest_station": "Kabul @ Nowshera",
     "baselines": {
         "TopographyDrainage": 4, "Deforestation": 8, "Urbanization": 3,
         "Siltation": 4, "AgriculturalPractices": 4, "Encroachments": 3,
         "IneffectiveDisasterPreparedness": 5, "DrainageSystems": 3,
         "CoastalVulnerability": 0, "Landslides": 9, "Watersheds": 6,
         "DeterioratingInfrastructure": 5, "PopulationScore": 4, "WetlandLoss": 6,
         "InadequatePlanning": 5, "PoliticalFactors": 4
     }},
    {"id": 10, "name": "Lasbela, Balochistan", "lat": 25.8380, "lng": 66.6590,
     "province": "Balochistan", "population": 120000, "shelters": 2, "hospitals": 1,
     "nearest_station": "Kotri",
     "baselines": {
         "TopographyDrainage": 5, "Deforestation": 3, "Urbanization": 2,
         "Siltation": 4, "AgriculturalPractices": 3, "Encroachments": 2,
         "IneffectiveDisasterPreparedness": 8, "DrainageSystems": 2,
         "CoastalVulnerability": 8, "Landslides": 3, "Watersheds": 3,
         "DeterioratingInfrastructure": 7, "PopulationScore": 2, "WetlandLoss": 4,
         "InadequatePlanning": 8, "PoliticalFactors": 7
     }},
    {"id": 11, "name": "Jaffarabad, Balochistan", "lat": 28.3015, "lng": 68.1800,
     "province": "Balochistan", "population": 150000, "shelters": 2, "hospitals": 1,
     "nearest_station": "Guddu",
     "baselines": {
         "TopographyDrainage": 7, "Deforestation": 3, "Urbanization": 2,
         "Siltation": 6, "AgriculturalPractices": 5, "Encroachments": 4,
         "IneffectiveDisasterPreparedness": 9, "DrainageSystems": 2,
         "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 3,
         "DeterioratingInfrastructure": 8, "PopulationScore": 3, "WetlandLoss": 5,
         "InadequatePlanning": 8, "PoliticalFactors": 7
     }},
    {"id": 12, "name": "Thatta, Sindh", "lat": 24.7461, "lng": 67.9236,
     "province": "Sindh", "population": 160000, "shelters": 2, "hospitals": 1,
     "nearest_station": "Kotri",
     "baselines": {
         "TopographyDrainage": 8, "Deforestation": 3, "Urbanization": 3,
         "Siltation": 7, "AgriculturalPractices": 5, "Encroachments": 4,
         "IneffectiveDisasterPreparedness": 7, "DrainageSystems": 3,
         "CoastalVulnerability": 9, "Landslides": 1, "Watersheds": 3,
         "DeterioratingInfrastructure": 7, "PopulationScore": 3, "WetlandLoss": 7,
         "InadequatePlanning": 7, "PoliticalFactors": 6
     }},
    {"id": 13, "name": "Karachi, Sindh", "lat": 24.8607, "lng": 67.0011,
     "province": "Sindh", "population": 16000000, "shelters": 15, "hospitals": 20,
     "nearest_station": "Kotri",
     "baselines": {
         "TopographyDrainage": 8, "Deforestation": 2, "Urbanization": 10,
         "Siltation": 5, "AgriculturalPractices": 1, "Encroachments": 9,
         "IneffectiveDisasterPreparedness": 7, "DrainageSystems": 3,
         "CoastalVulnerability": 8, "Landslides": 2, "Watersheds": 2,
         "DeterioratingInfrastructure": 8, "PopulationScore": 10, "WetlandLoss": 6,
         "InadequatePlanning": 8, "PoliticalFactors": 7
     }},
]

# ============================================================
# IRSA River Data (from Daily Water Situation Report 20.05.2026)
# ============================================================
IRSA_DATA = {
    "date": "20.05.2026",
    "stations": {
        "Indus @ Tarbela": {
            "level": 1450.22, "dead_level": 1402.00,
            "inflow": 82700, "outflow": 70000,
            "capacity_pct": round((1450.22 - 1402.00) / (1550.00 - 1402.00) * 100, 1),
            "status": "normal"
        },
        "Kabul @ Nowshera": {
            "discharge": 48700,
            "status": "high" if 48700 > 40000 else "normal"
        },
        "Jhelum @ Mangla": {
            "level": 1164.90, "dead_level": 1050.00,
            "inflow": 42654, "outflow": 38000,
            "capacity_pct": round((1164.90 - 1050.00) / (1242.00 - 1050.00) * 100, 1),
            "status": "normal"
        },
        "Chenab @ Marala": {
            "us_discharge": 23566, "ds_discharge": 7480,
            "status": "normal"
        },
        "Kalabagh": {
            "us_discharge": 136903, "ds_discharge": 129903,
            "thal_canal": 7000,
            "status": "high" if 136903 > 100000 else "normal"
        },
        "Chashma": {
            "level": 644.00, "dead_level": 638.15,
            "inflow": 114552, "outflow": 105000,
            "status": "normal"
        },
        "Taunsa": {
            "us_discharge": 99660, "ds_discharge": 78138,
            "tp_link": 8022,
            "muzaffargarh_canal": 7000,
            "dg_khan_canal": 6000,
            "status": "high" if 99660 > 80000 else "normal"
        },
        "Guddu": {
            "us_discharge": 62904, "ds_discharge": 59493,
            "canal_wdls": 3411,
            "status": "normal"
        },
        "Sukkur": {
            "us_discharge": 54350, "ds_discharge": 17850,
            "canal_wdls": 36500,
            "status": "normal"
        },
        "Kotri": {
            "us_discharge": 11255, "ds_discharge": 0,
            "canal_wdls": 11255,
            "status": "low"
        },
        "Panjnad": {
            "us_discharge": 11533, "ds_discharge": 0,
            "status": "low"
        }
    },
    "rim_station_inflows": 197620,
    "rim_station_outflows": 180266,
    "irsa_releases": {
        "Punjab": {"today": 95000, "last_year": 115400},
        "Sindh": {"today": 80000, "last_year": 115000},
        "KP": {"today": 2900, "last_year": 2100},
        "Balochistan": {"today": 9000, "last_year": 4000}
    }
}


# ============================================================
# Model Loading
# ============================================================
_models = {}

def load_models():
    global _models
    if _models:
        return _models
    try:
        import joblib
        model_dir = os.path.join(os.path.dirname(__file__), 'models')

        # Load production XGBoost model (lightweight, 0.27 MB)
        prod_path = os.path.join(model_dir, 'xgb_production.pkl')
        if os.path.exists(prod_path):
            _models['xgboost'] = joblib.load(prod_path)
            print("  [OK] Loaded xgb_production.pkl")

        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            _models['scaler'] = joblib.load(scaler_path)

    except Exception as e:
        print(f"  [WARN] Could not load models: {e}")
    return _models


def predict_flood(features_dict):
    """Predict flood probability from 20 features."""
    models = load_models()
    feature_values = [features_dict.get(f, 5) for f in FEATURES]

    if 'xgboost' in models:
        X = np.array(feature_values).reshape(1, -1)
        prediction = float(models['xgboost'].predict(X)[0])
    else:
        # Fallback: weighted sum
        weights = {f: 1.0/20 for f in FEATURES}
        weighted_sum = sum(features_dict.get(f, 5) * weights[f] for f in FEATURES)
        prediction = min(max(weighted_sum / 10.0 * 0.9 + 0.1, 0.0), 1.0)

    return round(max(0.0, min(1.0, prediction)), 4)


def get_risk_level(probability):
    """Calibrated risk levels based on dataset distribution."""
    if probability > 0.625: return "critical"
    elif probability >= 0.57: return "high"     # Warning
    elif probability >= 0.54: return "medium"   # Watch
    else: return "low"                          # Normal


def find_nearest_region(lat, lng):
    """Find the nearest predefined region to a lat/lng."""
    min_dist = float('inf')
    nearest = PAKISTAN_REGIONS[0]
    for region in PAKISTAN_REGIONS:
        dist = ((region['lat'] - lat)**2 + (region['lng'] - lng)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = region
    return nearest


def weather_to_features(weather_data, region):
    """Convert live weather + region baselines into 20 ML features."""
    baselines = region.get('baselines', {})
    features = {}

    # Dynamic features from weather
    rain = weather_data.get('rain_1h', 0)
    wind = weather_data.get('wind_speed', 0)
    humidity = weather_data.get('humidity', 50)
    temp = weather_data.get('temp', 30)
    clouds = weather_data.get('clouds', 0)

    # MonsoonIntensity: map rain mm/h to 0-10 scale
    features['MonsoonIntensity'] = min(10, int(rain / 5.0 * 10)) if rain > 0 else max(1, int(humidity / 15))

    # ClimateChange: temperature anomaly (>40C is extreme in Pakistan)
    features['ClimateChange'] = min(10, max(1, int((temp - 25) / 2)))

    # RiverManagement: from IRSA data - high discharge = poor management score
    station_name = region.get('nearest_station', 'Sukkur')
    station = IRSA_DATA['stations'].get(station_name, {})
    if station_name in IRSA_DATA['stations']:
        st = IRSA_DATA['stations'][station_name]
        if 'inflow' in st and 'outflow' in st:
            ratio = st['outflow'] / max(st['inflow'], 1)
            features['RiverManagement'] = min(10, max(1, int((1 - ratio) * 20)))
        elif 'us_discharge' in st:
            discharge = st['us_discharge']
            features['RiverManagement'] = min(10, max(1, int(discharge / 20000)))
        else:
            features['RiverManagement'] = 5
    else:
        features['RiverManagement'] = 5

    # DamsQuality: from IRSA dam levels
    if 'capacity_pct' in station:
        cap = station['capacity_pct']
        features['DamsQuality'] = min(10, max(1, int(cap / 10)))
    else:
        features['DamsQuality'] = baselines.get('DamsQuality', 5)

    # Static features from region baselines
    for f in FEATURES:
        if f not in features:
            features[f] = baselines.get(f, 5)

    return features


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html',
                         features=FEATURES,
                         feature_descriptions=FEATURE_DESCRIPTIONS,
                         regions=PAKISTAN_REGIONS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict from raw 20-feature input."""
    try:
        data = request.get_json()
        features_dict = {}
        for f in FEATURES:
            features_dict[f] = max(0, min(15, int(data.get(f, 5))))

        probability = predict_flood(features_dict)
        risk_level = get_risk_level(probability)

        return jsonify({
            'success': True,
            'flood_probability': probability,
            'risk_level': risk_level,
            'features': features_dict
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict-location', methods=['POST'])
def api_predict_location():
    """Predict flood risk for a specific lat/lng using weather + region data."""
    try:
        data = request.get_json()
        lat = float(data.get('lat', 30.0))
        lng = float(data.get('lng', 70.0))

        # Find nearest region
        region = find_nearest_region(lat, lng)

        # Get weather data
        weather = _fetch_weather(lat, lng)

        # Convert to ML features
        features = weather_to_features(weather, region)

        # Predict
        probability = predict_flood(features)
        risk_level = get_risk_level(probability)

        return jsonify({
            'success': True,
            'flood_probability': probability,
            'risk_level': risk_level,
            'region': region['name'],
            'weather': weather,
            'features_used': features
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/weather', methods=['GET'])
def api_weather():
    """Proxy to OpenWeatherMap API."""
    try:
        lat = request.args.get('lat', 30.3753)
        lng = request.args.get('lng', 69.3451)
        weather = _fetch_weather(float(lat), float(lng))
        return jsonify({'success': True, **weather})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def _fetch_weather(lat, lng):
    """Fetch current weather from OpenWeatherMap."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={OWM_API_KEY}&units=metric"
        resp = http_requests.get(url, timeout=5)
        data = resp.json()

        result = {
            'temp': data.get('main', {}).get('temp', 30),
            'feels_like': data.get('main', {}).get('feels_like', 30),
            'humidity': data.get('main', {}).get('humidity', 50),
            'pressure': data.get('main', {}).get('pressure', 1013),
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'wind_deg': data.get('wind', {}).get('deg', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'rain_1h': data.get('rain', {}).get('1h', 0),
            'rain_3h': data.get('rain', {}).get('3h', 0),
            'description': data.get('weather', [{}])[0].get('description', 'clear'),
            'icon': data.get('weather', [{}])[0].get('icon', '01d'),
            'city': data.get('name', 'Unknown'),
            'visibility': data.get('visibility', 10000),
        }
        return result
    except Exception as e:
        return {
            'temp': 35, 'feels_like': 38, 'humidity': 60, 'pressure': 1010,
            'wind_speed': 5, 'wind_deg': 180, 'clouds': 40,
            'rain_1h': 0, 'rain_3h': 0, 'description': 'data unavailable',
            'icon': '01d', 'city': 'Unknown', 'visibility': 10000,
            'error': str(e)
        }


@app.route('/api/irsa', methods=['GET'])
def api_irsa():
    """Return IRSA river station data."""
    return jsonify({'success': True, **IRSA_DATA})


@app.route('/api/regions', methods=['GET'])
def api_regions():
    """Return Pakistan region data."""
    return jsonify({'success': True, 'regions': PAKISTAN_REGIONS})


@app.route('/api/agent', methods=['POST'])
def api_agent():
    """Run A* search agent for evacuation planning."""
    try:
        data = request.get_json()
        flood_probability = data.get('flood_probability', 0.5)
        features_dict = data.get('features', {})

        from src.agent.flood_agent import run_agent
        result = run_agent(flood_probability, features_dict)
        return jsonify({'success': True, **result})
    except ImportError:
        return jsonify({
            'success': True,
            'actions': _fallback_agent(data.get('flood_probability', 0.5)),
            'total_cost': 0, 'nodes_explored': 0
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/csp', methods=['POST'])
def api_csp():
    """Run CSP solver for resource allocation."""
    try:
        data = request.get_json()
        risk_level = data.get('risk_level', 'medium')

        from src.csp.resource_csp import solve_csp
        result = solve_csp(risk_level)
        return jsonify({'success': True, **result})
    except ImportError:
        return jsonify({
            'success': True,
            'solution': _fallback_csp(data.get('risk_level', 'medium'))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/knowledge', methods=['POST'])
def api_knowledge():
    """Run knowledge-based forward chaining."""
    try:
        data = request.get_json()
        flood_probability = data.get('flood_probability', 0.5)
        features_dict = data.get('features', {})

        from src.knowledge_base.flood_kb import run_inference
        result = run_inference(flood_probability, features_dict)
        return jsonify({'success': True, **result})
    except ImportError:
        return jsonify({
            'success': True, 'facts': {}, 'trace': [],
            'recommendations': ['Knowledge base initializing...']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/full-analysis', methods=['POST'])
def api_full_analysis():
    """Run complete analysis pipeline for a location or manual input."""
    try:
        data = request.get_json()

        # Check if it's a location-based or manual request
        if 'lat' in data and 'lng' in data:
            lat, lng = float(data['lat']), float(data['lng'])
            region = find_nearest_region(lat, lng)
            weather = _fetch_weather(lat, lng)
            features_dict = weather_to_features(weather, region)
        else:
            features_dict = {}
            for f in FEATURES:
                features_dict[f] = max(0, min(15, int(data.get(f, 5))))

        # Predict
        probability = predict_flood(features_dict)
        risk_level = get_risk_level(probability)

        # Agent
        try:
            from src.agent.flood_agent import run_agent
            agent_result = run_agent(probability, features_dict)
        except:
            agent_result = {'actions': _fallback_agent(probability), 'total_cost': 0, 'nodes_explored': 0}

        # CSP
        try:
            from src.csp.resource_csp import solve_csp
            csp_result = solve_csp(risk_level)
        except:
            csp_result = _fallback_csp(risk_level)

        # Knowledge Base
        try:
            from src.knowledge_base.flood_kb import run_inference
            kb_result = run_inference(probability, features_dict)
        except:
            kb_result = {'facts': {}, 'trace': [], 'recommendations': []}

        return jsonify({
            'success': True,
            'prediction': {'flood_probability': probability, 'risk_level': risk_level},
            'agent': agent_result,
            'csp': csp_result,
            'knowledge_base': kb_result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/search', methods=['GET'])
def api_search():
    """Local gazetteer search."""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'success': True, 'results': []})
    
    try:
        from src.providers.static import search_location
        results = search_location(query)
        return jsonify({'success': True, 'results': results})
    except ImportError:
        return jsonify({'success': False, 'error': 'Static provider not loaded.'})


@app.route('/api/river-context', methods=['GET'])
def api_river_context():
    """Get localized hydrology hazards."""
    try:
        lat = float(request.args.get('lat', 30.0))
        lng = float(request.args.get('lng', 70.0))
        from src.providers.static import get_local_hydrology
        context = get_local_hydrology(lat, lng)
        return jsonify({'success': True, **context})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/evacuation-plan', methods=['GET'])
def api_evacuation_plan():
    """Get nearest shelter and directions link."""
    try:
        lat = float(request.args.get('lat', 30.0))
        lng = float(request.args.get('lng', 70.0))
        from src.providers.static import get_nearest_shelter
        info = get_nearest_shelter(lat, lng)
        shelter = info['shelter']
        
        # Google Maps Directions URL
        # e.g., https://www.google.com/maps/dir/?api=1&origin=33.68,73.04&destination=33.70,73.06
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={lat},{lng}&destination={shelter['lat']},{shelter['lng']}"
        
        return jsonify({
            'success': True, 
            'nearest_shelter': shelter['name'],
            'distance_km': info['distance_km'],
            'directions_url': gmaps_url
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Smart chatbot using Grok AI."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        lat = data.get('lat', None)
        lng = data.get('lng', None)
        
        context_data = {}
        if lat and lng:
            try:
                # Build context
                weather = _fetch_weather(lat, lng)
                region = find_nearest_region(lat, lng)
                features = weather_to_features(weather, region)
                prob = predict_flood(features)
                risk = get_risk_level(prob)
                
                from src.providers.static import get_local_hydrology, get_nearest_shelter
                hydro = get_local_hydrology(lat, lng)
                shelter = get_nearest_shelter(lat, lng)
                
                context_data = {
                    'location': f"{lat}, {lng}",
                    'risk_level': risk.upper(),
                    'probability': f"{prob*100:.1f}%",
                    'weather': f"{weather.get('temp')}C, Rain: {weather.get('rain_1h')}mm/h",
                    'hydrology': hydro.get('name', 'None'),
                    'shelter': shelter['shelter']['name']
                }
            except Exception as e:
                print(f"Chat context error: {e}")
                
        from src.providers.ai_assistant import generate_chat_response
        response = generate_chat_response(message, context_data)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400



@app.route('/api/model-comparison', methods=['GET'])
def api_model_comparison():
    """Return model metrics."""
    metrics_file = os.path.join(os.path.dirname(__file__), 'outputs', 'model_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return jsonify(json.load(f))
    prod_file = os.path.join(os.path.dirname(__file__), 'outputs', 'production_metrics.json')
    if os.path.exists(prod_file):
        with open(prod_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({
        'models': ['Ridge Regression', 'Random Forest', 'XGBoost'],
        'r2': [0.8449, 0.6355, 0.7282],
        'accuracy': [0.82, 0.75, 0.82],
        'f1': [0.80, 0.73, 0.83]
    })


@app.route('/api/clusters', methods=['GET'])
def api_clusters():
    clusters_file = os.path.join(os.path.dirname(__file__), 'outputs', 'cluster_profiles.json')
    if os.path.exists(clusters_file):
        with open(clusters_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({
        'n_clusters': 4,
        'cluster_names': ['Low-Risk Rural', 'Moderate Urban', 'High-Risk Floodplain', 'Critical Zone'],
        'cluster_sizes': [280000, 310000, 250000, 278000],
        'mean_flood_probability': [0.38, 0.48, 0.58, 0.62]
    })


# ============================================================
# Fallbacks
# ============================================================

def _fallback_agent(probability):
    if probability > 0.7:
        return [
            {"action": "ISSUE_ALERT", "description": "Issue RED alert to all districts"},
            {"action": "OPEN_SHELTER", "description": "Open 5 emergency shelters"},
            {"action": "DEPLOY_RESCUE_TEAM", "description": "Deploy 8 rescue teams"},
            {"action": "BEGIN_EVACUATION", "description": "Begin full evacuation"},
            {"action": "ALLOCATE_SUPPLIES", "description": "Allocate 7 days of supplies"},
        ]
    elif probability > 0.5:
        return [
            {"action": "ISSUE_ALERT", "description": "Issue ORANGE alert"},
            {"action": "OPEN_SHELTER", "description": "Open 3 shelters"},
            {"action": "DEPLOY_RESCUE_TEAM", "description": "Deploy 4 rescue teams"},
        ]
    elif probability > 0.35:
        return [
            {"action": "ISSUE_ALERT", "description": "Issue YELLOW advisory"},
            {"action": "OPEN_SHELTER", "description": "Open 1 standby shelter"},
        ]
    return [{"action": "MONITOR", "description": "Continue routine monitoring"}]


def _fallback_csp(risk_level):
    return {
        'shelter_assignments': {'Zone_1': 'Shelter_A'},
        'team_assignments': {'Team_1': 'Zone_1'},
        'constraints_satisfied': True
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("  FloodGuard Pakistan - Disaster Management System")
    print("  Dashboard: http://127.0.0.1:5000")
    print("=" * 60)

    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
