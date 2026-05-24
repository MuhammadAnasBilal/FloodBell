from fastapi import APIRouter
import json
import os

router = APIRouter()

@router.get("/irsa")
async def get_irsa_data():
    """
    Returns the latest scraped IRSA data from the cache.
    """
    cache_file = os.path.join(os.path.dirname(__file__), '../../data/irsa_cache.json')
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Fallback if no scrape has happened yet
    return {
        "success": True,
        "date": "20.05.2026",
        "stations": {
            "Indus @ Tarbela": {"level": 1450.22, "inflow": 82700, "outflow": 70000},
            "Kabul @ Nowshera": {"discharge": 48700},
            "Jhelum @ Mangla": {"level": 1164.90, "inflow": 42654, "outflow": 38000},
            "Sukkur": {"us_discharge": 54350, "ds_discharge": 17850},
            "Kotri": {"us_discharge": 11255, "ds_discharge": 0}
        },
        "rim_station_inflows": 197620,
        "rim_station_outflows": 180266,
        "source": "Fallback Cache"
    }
