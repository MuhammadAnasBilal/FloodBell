import math

# Local Pakistan Gazetteer for Search Autocomplete
PAKISTAN_GAZETTEER = [
    {"name": "Islamabad", "lat": 33.6844, "lng": 73.0479, "type": "City"},
    {"name": "G-11/2, Islamabad", "lat": 33.6698, "lng": 72.9976, "type": "Sector"},
    {"name": "Sargodha", "lat": 32.0836, "lng": 72.6711, "type": "City"},
    {"name": "Karachi", "lat": 24.8607, "lng": 67.0011, "type": "City"},
    {"name": "Lahore", "lat": 31.5204, "lng": 74.3587, "type": "City"},
    {"name": "Peshawar", "lat": 34.0151, "lng": 71.5249, "type": "City"},
    {"name": "Quetta", "lat": 30.1798, "lng": 66.9750, "type": "City"},
    {"name": "Sukkur", "lat": 27.7052, "lng": 68.8574, "type": "City"},
    {"name": "Nowshera", "lat": 34.0153, "lng": 71.9747, "type": "City"},
    {"name": "Dadu", "lat": 26.7319, "lng": 67.7750, "type": "City"},
    {"name": "Jacobabad", "lat": 28.2769, "lng": 68.4514, "type": "City"},
    {"name": "Charsadda", "lat": 34.1478, "lng": 71.7310, "type": "City"},
    {"name": "Dera Ghazi Khan", "lat": 30.0561, "lng": 70.6346, "type": "City"},
    {"name": "Rajanpur", "lat": 29.1044, "lng": 70.3301, "type": "City"},
    {"name": "Muzaffargarh", "lat": 30.0740, "lng": 71.1932, "type": "City"},
    {"name": "Swat", "lat": 35.2227, "lng": 72.3528, "type": "City"},
    {"name": "Lasbela", "lat": 25.8380, "lng": 66.6590, "type": "City"},
    {"name": "Jaffarabad", "lat": 28.3015, "lng": 68.1800, "type": "City"},
    {"name": "Thatta", "lat": 24.7461, "lng": 67.9236, "type": "City"}
]

# Local Hydrology mapping - Maps coordinates to specific localized hazards
HYDROLOGY_ZONES = [
    {
        "name": "Nala Lai / Rawal Basin",
        "bounds": {"lat_min": 33.5, "lat_max": 33.8, "lng_min": 72.9, "lng_max": 73.2},
        "description": "Urban drainage hazard. Prone to flash flooding during intense monsoon spells.",
        "hazards": ["Nala Lai", "Rawal Dam Spillway", "Urban Street Flooding"]
    },
    {
        "name": "Sargodha Plains / Jhelum River",
        "bounds": {"lat_min": 31.8, "lat_max": 32.3, "lng_min": 72.3, "lng_max": 72.9},
        "description": "Riverine flood zone. Affected by Jhelum River overflows and heavy plains rain.",
        "hazards": ["Jhelum River", "Local Canal Breaches"]
    },
    {
        "name": "Kabul River Basin",
        "bounds": {"lat_min": 33.8, "lat_max": 34.3, "lng_min": 71.5, "lng_max": 72.2},
        "description": "High risk of sudden river swelling from Kabul and Swat rivers.",
        "hazards": ["Kabul River", "Swat River Confluence", "Warsak Dam releases"]
    },
    {
        "name": "Lower Indus Basin (Sukkur to Kotri)",
        "bounds": {"lat_min": 24.5, "lat_max": 28.5, "lng_min": 67.5, "lng_max": 69.0},
        "description": "Major riverine flooding path. Wide plains and barrage bottlenecks.",
        "hazards": ["Indus River Mainstream", "Sukkur Barrage", "Kotri Barrage", "Canal Overflows"]
    },
    {
        "name": "Karachi Coastal / Urban",
        "bounds": {"lat_min": 24.7, "lat_max": 25.1, "lng_min": 66.8, "lng_max": 67.2},
        "description": "Urban drainage crisis point and coastal storm surge vulnerability.",
        "hazards": ["Malir River", "Lyari River", "Urban Choke Points", "Coastal Surge"]
    }
]

def search_location(query):
    query = query.lower()
    results = [p for p in PAKISTAN_GAZETTEER if query in p['name'].lower()]
    return results[:5]

def get_local_hydrology(lat, lng):
    """Finds localized hydrology contexts based on coordinates."""
    contexts = []
    for zone in HYDROLOGY_ZONES:
        b = zone['bounds']
        if b['lat_min'] <= lat <= b['lat_max'] and b['lng_min'] <= lng <= b['lng_max']:
            contexts.append(zone)
            
    if not contexts:
        return {"name": "No major nearby river hazard", "hazards": [], "description": "Area primarily subject to local rainfall accumulation."}
        
    return contexts[0]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_nearest_shelter(lat, lng):
    # Dummy logic for nearest shelter - in a real app, this would query a database of shelters
    shelters = [
        {"name": "Islamabad Sports Complex Shelter", "lat": 33.70, "lng": 73.06},
        {"name": "Sargodha Stadium Camp", "lat": 32.08, "lng": 72.67},
        {"name": "Nowshera Degree College", "lat": 34.01, "lng": 71.98},
        {"name": "Sukkur IBA Safe Zone", "lat": 27.72, "lng": 68.83},
        {"name": "Karachi Expo Center", "lat": 24.89, "lng": 67.07}
    ]
    
    nearest = min(shelters, key=lambda x: haversine(lat, lng, x['lat'], x['lng']))
    distance = haversine(lat, lng, nearest['lat'], nearest['lng'])
    return {"shelter": nearest, "distance_km": round(distance, 1)}
