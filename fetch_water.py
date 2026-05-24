import requests
import json
import os

overpass_url = "http://overpass-api.de/api/interpreter"

# Query for major rivers and reservoirs in Pakistan
overpass_query = """
[out:json][timeout:60];
area["name:en"="Pakistan"]->.searchArea;
(
  way["waterway"="river"](area.searchArea);
  way["water"="river"](area.searchArea);
  way["water"="reservoir"](area.searchArea);
  relation["waterway"="river"](area.searchArea);
  relation["water"="reservoir"](area.searchArea);
);
out geom;
"""

print("Fetching water bodies data from Overpass API (this may take a minute)...")
try:
    response = requests.post(overpass_url, data={'data': overpass_query}, timeout=70)
    response.raise_for_status()
    data = response.json()
    
    # Convert Overpass JSON to GeoJSON
    features = []
    for element in data.get('elements', []):
        if element['type'] == 'way' and 'geometry' in element:
            coords = [[node['lon'], node['lat']] for node in element['geometry']]
            features.append({
                "type": "Feature",
                "properties": element.get('tags', {}),
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            })
            
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    out_path = os.path.join("frontend", "public", "pakistan_water_bodies.geojson")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
        
    print(f"Successfully saved {len(features)} water bodies to {out_path}")
except Exception as e:
    print(f"Failed to fetch data: {e}")
    # Create an empty geojson so the map doesn't crash
    empty_geojson = {"type": "FeatureCollection", "features": []}
    out_path = os.path.join("frontend", "public", "pakistan_water_bodies.geojson")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(empty_geojson, f)
