import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap, useMapEvents, GeoJSON } from 'react-leaflet';
import L from 'leaflet';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, MapPin, AlertTriangle, Info, MessageCircle, Layers, X, Send, Droplets, Wind, Thermometer, CloudRain, CloudLightning, Activity, Cloud, Navigation, ChevronLeft, ChevronRight, Menu } from 'lucide-react';
import VelocityLayer from '../components/VelocityLayer';

// Fix Leaflet's default icon path issues
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// --- Leaflet Logic Components ---

function MapController({ position, setPosition, setMapInstance, isEmergency }) {
  const map = useMap();
  
  useEffect(() => {
    if (map) {
      setMapInstance(map);
      const pakistanBounds = L.latLngBounds(
        L.latLng(23.5, 60.5), // Southwest corner
        L.latLng(37.5, 78.0)  // Northeast corner
      );
      map.setMaxBounds(pakistanBounds);
      map.options.maxBoundsViscosity = 1.0; // Hard bounce, no dragging outside
      map.options.minZoom = 6;
      map.options.maxZoom = 18;

      map.locate().on("locationfound", function (e) {
        setPosition(e.latlng);
        map.flyTo(e.latlng, 13);
      });
    }
  }, [map, setPosition, setMapInstance]);

  useEffect(() => {
    if (position && map) {
      map.flyTo(position, 13);
    }
  }, [position, map]);

  return position === null ? null : (
    <Marker 
      position={position} 
      draggable={true} 
      eventHandlers={{ 
        dragend: (e) => setPosition(e.target.getLatLng()),
        click: () => map.flyTo(position, 14) // Auto zoom in when clicked!
      }}
    >
      <Popup>Target Location</Popup>
      <Circle 
        center={position} 
        radius={10000} 
        pathOptions={{ 
          color: isEmergency ? 'red' : 'green', 
          fillColor: isEmergency ? '#ef4444' : '#22c55e', 
          fillOpacity: 0.15 
        }} 
      />
    </Marker>
  );
}

// Locate Me Button Control
function LocateControl({ setPosition }) {
  const map = useMap();
  return (
    <div className="absolute right-4 top-24 z-[1000]">
      <button 
        onClick={() => {
          map.locate().on("locationfound", function (e) {
            setPosition(e.latlng);
            map.flyTo(e.latlng, 13);
          });
        }}
        className="bg-white p-2 rounded-xl shadow-lg border border-gray-200 text-gray-700 hover:text-brand-primary hover:bg-blue-50 transition-colors"
        title="Go to my live location"
      >
        <Navigation size={20} className="transform -rotate-45" />
      </button>
    </div>
  );
}

// Mouse hover weather tooltip component
function MouseWeatherTooltip({ setHoverValue, activeLayer }) {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [latLng, setLatLng] = useState(null);
  const [weatherData, setWeatherData] = useState(null);

  useMapEvents({
    mousemove(e) {
      setMousePos({ x: e.originalEvent.clientX, y: e.originalEvent.clientY });
      setLatLng(e.latlng);
    }
  });

  useEffect(() => {
    if (!latLng) return;
    const timer = setTimeout(async () => {
      try {
        const apiKey = 'd1ccaf30c2e39ba82506f4ed9bdb50f2';
        const res = await fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${latLng.lat}&lon=${latLng.lng}&appid=${apiKey}&units=metric`);
        const data = await res.json();
        if (data.main) {
          setWeatherData(data);
          
          if (activeLayer === 'temp_new' || activeLayer === 'feels_like') setHoverValue(data.main.temp);
          else if (activeLayer === 'precipitation_new' || activeLayer === 'thunder') setHoverValue(data.rain ? data.rain['1h'] || 0 : 0);
          else if (activeLayer === 'wind_new') setHoverValue(data.wind.speed * 3.6); 
          else if (activeLayer === 'humidity') setHoverValue(data.main.humidity);
          else setHoverValue(data.clouds.all);
        }
      } catch (err) {}
    }, 150); 
    return () => clearTimeout(timer);
  }, [latLng, setHoverValue, activeLayer]);

  if (!latLng || !weatherData) return null;

  return (
    <div 
      className="fixed z-[9999] pointer-events-none bg-black/90 backdrop-blur-md text-white p-2.5 rounded-xl border border-white/20 shadow-2xl transition-transform duration-75"
      style={{ left: mousePos.x + 20, top: mousePos.y + 20 }}
    >
      <div className="absolute -top-2 -left-2 w-4 h-4 bg-black/90 border-t border-l border-white/20 transform rotate-45"></div>
      <div className="flex flex-col gap-1.5 text-xs font-bold relative z-10">
        <div className="flex items-center text-orange-400 text-sm"><Thermometer size={14} className="mr-1"/> {weatherData.main.temp.toFixed(1)}°C</div>
        <div className="flex items-center text-blue-400"><Droplets size={12} className="mr-1"/> {weatherData.main.humidity}% Humidity</div>
        <div className="flex items-center text-cyan-300"><Wind size={12} className="mr-1"/> {(weatherData.wind.speed * 3.6).toFixed(1)} km/h Wind</div>
      </div>
    </div>
  );
}

// Dynamic Legend with Multiple Value Stops
const Legend = ({ activeLayer, hoverValue }) => {
  let gradient = "";
  let min = 0, max = 100, unit = "";
  let stops = [];

  if (activeLayer === 'temp_new' || activeLayer === 'feels_like') {
    gradient = "linear-gradient(to right, #4a0082, #0000ff, #00ffff, #00ff00, #ffff00, #ff7f00, #ff0000, #800000)";
    min = -40; max = 50; unit = "°C";
    stops = [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50];
  } else if (activeLayer === 'precipitation_new' || activeLayer === 'thunder') {
    gradient = "linear-gradient(to right, transparent, #00ffff, #0000ff, #ff00ff, #ff0000)";
    min = 0; max = 140; unit = "mm";
    stops = [0, 5, 10, 20, 50, 100, 140];
  } else if (activeLayer === 'wind_new') {
    gradient = "linear-gradient(to right, transparent, #ffff00, #ff0000, #800080)";
    min = 0; max = 200; unit = "km/h";
    stops = [0, 20, 50, 100, 150, 200];
  } else {
    gradient = "linear-gradient(to right, transparent, #ffffff)";
    min = 0; max = 100; unit = "%";
    stops = [0, 25, 50, 75, 100];
  }

  let percentage = 0;
  if (hoverValue !== null) {
    percentage = ((hoverValue - min) / (max - min)) * 100;
    percentage = Math.max(0, Math.min(100, percentage));
  }

  return (
    <div className="absolute left-1/2 bottom-12 -translate-x-1/2 z-[1000] flex items-center bg-black/80 backdrop-blur-md px-6 py-3 rounded-full border border-white/20 shadow-2xl">
      <span className="text-white text-[11px] font-bold mr-4 w-8">{unit}</span>
      <div className="relative w-80 h-4 rounded-full" style={{ background: gradient }}>
        {/* Render labeled stops alongside the meter */}
        {stops.map((val, idx) => {
          let pos = ((val - min) / (max - min)) * 100;
          return (
            <div key={idx} className="absolute -top-6 text-[11px] text-white font-black drop-shadow-md transform -translate-x-1/2" style={{ left: `${pos}%` }}>
              {val}
            </div>
          );
        })}
        {/* Animated Pointer Arrow */}
        <motion.div 
          animate={{ left: `${percentage}%` }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="absolute -bottom-3 w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[8px] border-b-white transform -translate-x-1/2"
        />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [position, setPosition] = useState(null);
  const [mapInstance, setMapInstance] = useState(null);
  const [activeLayers, setActiveLayers] = useState([]); // Multiple layers
  const [windEnabled, setWindEnabled] = useState(true); // Wind is separate & always blowing
  const [baseLayer, setBaseLayer] = useState('terrain'); 
  const [baseMapMenuOpen, setBaseMapMenuOpen] = useState(false);
  const [hoverValue, setHoverValue] = useState(null);
  const [isEmergency, setIsEmergency] = useState(false); // Mock emergency state
  const [liveFloodRisk, setLiveFloodRisk] = useState(0.87); // ML value
  const [geoData, setGeoData] = useState(null);
  const [waterBodiesData, setWaterBodiesData] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [layersOpen, setLayersOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: "Hello! I am DeepSeek Flood AI. I am connected to the live ML model and IRSA river data. How can I assist you with your climate questions or evacuation plans?" }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const pakistanCenter = [30.3753, 69.3451];
  
  const layerOptions = [
    { id: 'temp_new', name: 'Temperature', icon: <Thermometer size={16} /> },
    { id: 'feels_like', name: 'Feels like temp', icon: <Thermometer size={16} className="text-orange-400" /> },
    { id: 'precipitation_new', name: 'Precipitation', icon: <CloudRain size={16} className="text-blue-400" /> },
    { id: 'clouds_new', name: 'Clouds', icon: <Cloud size={16} className="text-gray-400" /> },
    { id: 'wind_new', name: 'Wind speed', icon: <Wind size={16} className="text-cyan-400" /> },
    { id: 'pressure_new', name: 'Air pressure', icon: <Activity size={16} className="text-indigo-400" /> },
    { id: 'thunder', name: 'Thunderstorms', icon: <CloudLightning size={16} className="text-yellow-400" /> },
    { id: 'humidity', name: 'Humidity', icon: <Droplets size={16} className="text-blue-300" /> }
  ];

  useEffect(() => {
    fetch('/countries.geojson')
      .then(res => res.json())
      .then(data => {
        const pakistan = data.features.find(f => f.properties.ADMIN === 'Pakistan');
        if (pakistan) setGeoData(pakistan);
      });
      
    // Fetch newly downloaded rivers & dams dataset
    fetch('/pakistan_water_bodies.geojson')
      .then(res => res.json())
      .then(data => setWaterBodiesData(data))
      .catch(e => console.error("Water bodies GeoJSON not found yet."));
  }, []);

  useEffect(() => {
    if (searchQuery.length < 3) {
      setSuggestions([]);
      setIsSearching(false);
      return;
    }
    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${searchQuery}&countrycodes=pk&limit=5&addressdetails=1`);
        const data = await res.json();
        setSuggestions(data);
      } catch (err) {} finally {
        setIsSearching(false);
      }
    }, 800); // 800ms debounce to prevent API rate limiting
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const selectLocation = (lat, lon) => {
    setPosition(L.latLng(parseFloat(lat), parseFloat(lon)));
    setSearchQuery('');
    setSuggestions([]);
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput;
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const res = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, lat: position.lat, lng: position.lng })
      });
      const data = await res.json();
      if (data.success) {
        setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        setChatMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to DeepSeek API." }]);
      }
    } catch (e) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: "Error: Backend not running." }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  useEffect(() => {
    if (!position) return;
    const fetchPrediction = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/predict/location', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: position.lat, lng: position.lng })
        });
        const data = await res.json();
        if (data.success) {
          setLiveFloodRisk(data.flood_probability);
          setIsEmergency(data.flood_probability > 0.625);
        }
      } catch (err) {
        console.error("Backend not reachable");
      }
    };
    fetchPrediction();
  }, [position]);

  // Maps UI layer selection to underlying OpenWeatherMap standard tiles
  const getOwmLayerId = (layerId) => {
    if(layerId === 'feels_like') return 'temp_new';
    if(layerId === 'thunder') return 'precipitation_new';
    if(layerId === 'humidity') return 'precipitation_new'; // OWM free doesn't have humidity tile, fallback visually
    return layerId;
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-[#111] flex">
      
      {/* Solid Analytics Sidebar (Left) */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div 
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 340, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="h-full bg-gray-900 border-r border-white/10 shadow-2xl flex flex-col z-[1000] relative shrink-0"
          >
            {/* App Logo */}
            <div className="p-6 border-b border-white/10 flex items-center bg-black/40">
              <div className="mr-3">
                <img src="/logo_rounded.png" alt="Logo" className="w-10 h-10 rounded-[10px] shadow-[0_0_20px_rgba(59,130,246,0.5)] border border-white/20" />
              </div>
              <div>
                <h1 className="text-white font-black tracking-widest text-xl">FLOOD BELL</h1>
                <p className="text-[10px] text-gray-400 uppercase tracking-widest font-bold">Early Warning System</p>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-5 space-y-5 no-scrollbar">
              <div className="bg-black/50 border border-white/10 rounded-2xl p-5 shadow-inner">
                <h3 className="text-gray-400 text-[10px] font-bold uppercase tracking-widest mb-3 flex items-center"><AlertTriangle size={12} className="mr-1 text-red-500" /> Live Flood Risk</h3>
                <div className="flex justify-between items-end mb-2">
                  <span className={`${isEmergency ? 'text-red-500 animate-pulse' : 'text-blue-400'} font-black text-3xl leading-none`}>
                    {(liveFloodRisk * 100).toFixed(0)}%
                  </span>
                  <span className="text-gray-500 text-xs font-bold mb-1">Current Location</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-2">
                  <div className={`h-2 rounded-full ${isEmergency ? 'bg-red-600 shadow-[0_0_10px_rgba(220,38,38,0.8)]' : 'bg-gradient-to-r from-green-500 to-yellow-500'}`} style={{ width: `${Math.min(100, liveFloodRisk * 100)}%` }}></div>
                </div>
              </div>

              <div className="bg-black/50 border border-white/10 rounded-2xl p-5 shadow-inner">
                <h3 className="text-gray-400 text-[10px] font-bold uppercase tracking-widest mb-3 flex items-center">
                  <Info size={12} className="mr-1 text-blue-400" /> IRSA Daily Scraping
                </h3>
                <p className="text-[10px] text-yellow-500/90 mb-4 bg-yellow-500/10 p-2.5 rounded-lg border border-yellow-500/20 leading-relaxed">⚠️ <b>UI Placeholder.</b> The Python Backend pipeline is currently being built to scrape live data directly from IRSA servers.</p>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-gray-400 font-medium">Nearest Dam</span>
                    <span className="text-white font-bold">Tarbela Dam</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-gray-400 font-medium">Mean Inflow</span>
                    <span className="text-blue-400 font-mono font-bold">-- Cusecs</span>
                  </div>
                  <div className="flex justify-between pb-1">
                    <span className="text-gray-400 font-medium">Mean Outflow</span>
                    <span className="text-orange-400 font-mono font-bold">-- Cusecs</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Sidebar toggle button (when open) */}
            <button 
              onClick={() => setSidebarOpen(false)}
              className="absolute -right-4 top-1/2 -translate-y-1/2 bg-gray-900 border border-white/10 text-white p-1 rounded-full shadow-lg hover:text-brand-primary"
            >
              <ChevronLeft size={20} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Map Area */}
      <div className="flex-1 relative h-full">
        
        {/* Sidebar toggle button (when closed) */}
        {!sidebarOpen && (
          <button 
            onClick={() => setSidebarOpen(true)}
            className="absolute left-4 top-4 z-[1000] bg-gray-900/90 backdrop-blur-md border border-white/10 text-white p-3 rounded-xl shadow-2xl hover:bg-gray-800"
          >
            <Menu size={20} />
          </button>
        )}

        {/* Water Bodies Legend */}
        <div className="absolute top-20 left-4 z-[1000] bg-white border border-gray-200 p-4 rounded-xl shadow-[0_5px_15px_rgba(0,0,0,0.1)] flex flex-col gap-3">
          <h4 className="font-black text-gray-800 text-xs uppercase tracking-wider">Rivers & Lakes</h4>
          <div className="flex items-center text-xs font-bold text-gray-600"><div className="w-5 h-1 bg-blue-500 mr-3 rounded-full"></div> River</div>
          <div className="flex items-center text-xs font-bold text-gray-600">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="#f97316" stroke="#c2410c" strokeWidth="2" className="mr-3"><polygon points="12 2 22 20 2 20" /></svg> Dam
          </div>
          <div className="flex items-center text-xs font-bold text-gray-600">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="#22c55e" stroke="#166534" strokeWidth="2" className="mr-3"><circle cx="12" cy="12" r="10" /></svg> Barrage
          </div>
          
          <div className="mt-1 pt-2 border-t border-gray-100 flex items-center text-[10px] font-bold text-red-500">
            <span className="w-2.5 h-2.5 rounded-full bg-red-600 animate-pulse mr-2 border border-white shadow-md"></span> Spillway Open / Overflow
          </div>
        </div>

        {/* Ventusky-Style Search Bar & Locate Button */}
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] w-full max-w-xl px-4 flex items-center gap-3">
          <motion.div 
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="bg-white rounded-full shadow-[0_10px_30px_rgba(0,0,0,0.3)] flex items-center px-2 py-1.5 flex-1 relative z-20"
          >
            <div className="flex-1 flex items-center px-4">
              <input 
                type="text" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for Location..." 
                className="bg-transparent border-none text-gray-800 w-full focus:outline-none placeholder-gray-400 font-medium"
              />
              {isSearching ? (
                <div className="ml-2 w-5 h-5 border-2 border-brand-primary/30 border-t-brand-primary rounded-full animate-spin" />
              ) : (
                <Search size={20} className="text-brand-primary/80 ml-2" />
              )}
            </div>
            <button className="bg-brand-primary p-3 rounded-full text-white shadow-lg hover:bg-blue-600 transition-colors">
              <Navigation size={18} className="transform rotate-45 -translate-y-[1px] -translate-x-[1px]" />
            </button>
          </motion.div>

          {/* Locate Me Button Next to Search Bar */}
          <motion.button 
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            onClick={() => {
              if (mapInstance) {
                mapInstance.locate().on("locationfound", function (e) {
                  setPosition(e.latlng);
                  mapInstance.flyTo(e.latlng, 13);
                });
              }
            }}
            className="bg-white p-3.5 rounded-full shadow-[0_10px_30px_rgba(0,0,0,0.3)] text-gray-700 hover:text-brand-primary hover:bg-blue-50 transition-colors z-20 flex-shrink-0"
            title="Go to my live location"
          >
            <Navigation size={20} className="transform -rotate-45" />
          </motion.button>
          
          {/* Search Suggestions Dropdown */}
          <AnimatePresence>
            {suggestions.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-14 left-0 right-14 w-[calc(100%-3.5rem)] bg-white rounded-2xl shadow-2xl mt-2 overflow-hidden z-10"
              >
                {suggestions.map((item, i) => (
                  <div 
                    key={i} 
                    onClick={() => selectLocation(item.lat, item.lon)}
                    className="px-5 py-3 border-b border-gray-100 hover:bg-blue-50 cursor-pointer flex items-start"
                  >
                    <div className="text-orange-500 mr-3 mt-1"><MapPin size={16} /></div>
                    <div>
                      <div className="text-sm font-bold text-gray-800">{item.name || item.display_name.split(',')[0]}</div>
                      <div className="text-[11px] text-gray-500 font-medium mt-0.5">
                        Lat.: {parseFloat(item.lat).toFixed(2)}°N / Lon.: {parseFloat(item.lon).toFixed(2)}°E / {item.address?.state || "Pakistan"}
                      </div>
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Global Wind Footer (Ventusky style) */}
        <div className="absolute bottom-0 left-0 w-full h-8 bg-black/60 backdrop-blur-md z-[999] flex items-center px-6 text-white text-[11px] font-bold border-t border-white/10">
          Global Wind Direction: <span className="text-brand-primary mx-1">137.47°</span> , Global Wind Speed: <span className="text-brand-primary mx-1">7.39 m/s</span>
          <span className="ml-auto text-gray-400 font-normal">⚠️ Wind particles using sample JSON. Requires Python Backend for live GFS data.</span>
        </div>

        {/* Slide-out Layer Menu (Right) */}
        <motion.div 
          initial={{ x: 300 }}
          animate={{ x: layersOpen ? 0 : 300 }}
          className="absolute top-24 right-0 z-[1000] h-auto max-h-[70vh] w-64 bg-white border-l border-y border-gray-200 rounded-l-2xl shadow-2xl flex flex-col"
        >
          <button 
            onClick={() => setLayersOpen(!layersOpen)}
            className="absolute -left-auto right-[100%] top-4 bg-white px-4 py-3 rounded-l-full border-y border-l border-gray-200 text-gray-800 flex items-center font-bold shadow-[-5px_5px_15px_rgba(0,0,0,0.1)] hover:bg-gray-50 transition-colors whitespace-nowrap min-w-[135px]"
          >
            <span className="mr-2 text-brand-primary"><Layers size={16} /></span>
            Map Layers
          </button>
          
          <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
            <h3 className="text-gray-800 font-bold tracking-wide">Weather Overlays</h3>
            <X size={16} className="text-gray-400 cursor-pointer hover:text-red-500" onClick={() => setLayersOpen(false)} />
          </div>
          <div className="overflow-y-auto p-4 space-y-3 no-scrollbar flex-1">
            
            {/* Dedicated Wind Toggle */}
            <div className="flex items-center justify-between">
              <div className="flex items-center text-sm font-bold text-gray-700">
                <Wind size={16} className="text-cyan-400 mr-3" /> Live Wind Flow
              </div>
              <button 
                onClick={() => setWindEnabled(!windEnabled)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${windEnabled ? 'bg-green-500' : 'bg-gray-300'}`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${windEnabled ? 'translate-x-6' : 'translate-x-1'}`} />
              </button>
            </div>
            
            <div className="h-px bg-gray-100 my-2" />

            {/* Weather Overlay Toggles (Excluding Wind) */}
            {layerOptions.filter(l => l.id !== 'wind_new').map(layer => (
              <div key={layer.id} className="flex items-center justify-between">
                <div className="flex items-center text-sm font-bold text-gray-700">
                  <span className="mr-3">{layer.icon}</span> {layer.name}
                </div>
                <button 
                  onClick={() => {
                    if (activeLayers.includes(layer.id)) {
                      setActiveLayers(activeLayers.filter(id => id !== layer.id));
                    } else {
                      setActiveLayers([...activeLayers, layer.id]);
                    }
                  }}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${activeLayers.includes(layer.id) ? 'bg-orange-500' : 'bg-gray-300'}`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${activeLayers.includes(layer.id) ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Dynamic Color Scale Legend */}
        <Legend activeLayer={activeLayers[0]} hoverValue={hoverValue} />

        {/* Floating DeepSeek AI Chatbot */}
        <div className="absolute bottom-12 right-6 z-[1000] flex flex-col items-end">
          <AnimatePresence>
            {chatOpen && (
              <motion.div 
                initial={{ opacity: 0, y: 20, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 20, scale: 0.9 }}
                className="bg-gray-900 border border-white/10 rounded-3xl shadow-[0_10px_40px_rgba(0,0,0,0.5)] w-80 h-96 mb-4 flex flex-col overflow-hidden backdrop-blur-2xl"
              >
                <div className="bg-gradient-to-r from-brand-primary to-brand-accent p-4 flex justify-between items-center text-white">
                  <div className="flex items-center"><MessageCircle size={18} className="mr-2" /> <span className="font-bold text-sm tracking-wide">DeepSeek Flood AI</span></div>
                  <X size={18} className="cursor-pointer hover:text-gray-200 transition-colors" onClick={() => setChatOpen(false)} />
                </div>
                <div className="flex-1 p-4 overflow-y-auto bg-black/60 text-sm space-y-4 no-scrollbar">
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`p-3.5 rounded-2xl max-w-[90%] shadow-md leading-relaxed ${msg.role === 'user' ? 'bg-brand-primary text-white self-end rounded-tr-none ml-auto' : 'bg-gray-800 text-gray-200 rounded-tl-none border border-white/5 mr-auto'}`}>
                      {msg.content}
                    </div>
                  ))}
                  {isChatLoading && (
                    <div className="bg-gray-800 text-gray-200 p-3.5 rounded-2xl rounded-tl-none self-start w-16 border border-white/5 flex items-center justify-center space-x-1">
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  )}
                </div>
                <div className="p-3 bg-gray-900 border-t border-white/10 flex items-center">
                  <input 
                    type="text" 
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask about evacuation..." 
                    className="flex-1 bg-black/50 border border-white/10 text-white text-sm focus:outline-none focus:border-brand-primary rounded-full px-4 py-2.5 transition-colors" 
                  />
                  <button onClick={handleSendMessage} className="text-white bg-brand-primary hover:bg-blue-500 p-2.5 rounded-full ml-2 shadow-lg transition-colors">
                    <Send size={16} />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setChatOpen(!chatOpen)}
            className="w-14 h-14 bg-brand-primary rounded-full shadow-[0_0_20px_rgba(59,130,246,0.6)] flex items-center justify-center text-white relative border-2 border-white/20"
          >
            {chatOpen ? <X size={24} /> : <MessageCircle size={24} />}
            {!chatOpen && <span className="absolute top-0 right-0 w-3 h-3 bg-red-500 rounded-full border-2 border-[#111] animate-pulse"></span>}
          </motion.button>
        </div>

        {/* Fullscreen Map */}
        <MapContainer 
          center={pakistanCenter} 
          zoom={6} 
          className="w-full h-full z-0 cursor-crosshair"
          zoomControl={false}
        >
          <MapController position={position} setPosition={setPosition} setMapInstance={setMapInstance} isEmergency={isEmergency} />
        
        <MouseWeatherTooltip setHoverValue={setHoverValue} activeLayer={activeLayers[0]} />

        {/* Separate Base Map Layers Popup Button */}
        <div className="absolute bottom-12 left-6 z-[1000] flex flex-col items-start">
          <AnimatePresence>
            {baseMapMenuOpen && (
              <motion.div 
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="bg-white border border-gray-200 rounded-2xl shadow-2xl w-48 mb-3 overflow-hidden flex flex-col"
              >
                <div className="p-3 border-b border-gray-100 bg-gray-50 flex justify-between items-center">
                  <span className="text-xs font-bold text-gray-700">Map Type</span>
                  <X size={14} className="cursor-pointer text-gray-400 hover:text-red-500" onClick={() => setBaseMapMenuOpen(false)} />
                </div>
                <div className="p-1.5">
                  <button 
                    onClick={() => { setBaseLayer('satellite'); setBaseMapMenuOpen(false); }}
                    className={`w-full px-3 py-2 text-xs text-left rounded-lg font-bold transition-colors ${baseLayer === 'satellite' ? 'bg-blue-50 text-brand-primary' : 'text-gray-600 hover:bg-gray-50'}`}
                  >
                    🛰️ Satellite View
                  </button>
                  <button 
                    onClick={() => { setBaseLayer('terrain'); setBaseMapMenuOpen(false); }}
                    className={`w-full px-3 py-2 text-xs text-left rounded-lg font-bold transition-colors ${baseLayer === 'terrain' ? 'bg-blue-50 text-brand-primary' : 'text-gray-600 hover:bg-gray-50'}`}
                  >
                    🗺️ Light Terrain View
                  </button>
                  <button 
                    onClick={() => { setBaseLayer('street'); setBaseMapMenuOpen(false); }}
                    className={`w-full px-3 py-2 text-xs text-left rounded-lg font-bold transition-colors ${baseLayer === 'street' ? 'bg-blue-50 text-brand-primary' : 'text-gray-600 hover:bg-gray-50'}`}
                  >
                    🛣️ Street Map View
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setBaseMapMenuOpen(!baseMapMenuOpen)}
            className="w-12 h-12 bg-white rounded-xl shadow-[0_5px_15px_rgba(0,0,0,0.2)] flex items-center justify-center text-gray-700 border border-gray-200 hover:text-brand-primary transition-colors"
            title="Change Base Map"
          >
            <Layers size={22} />
          </motion.button>
        </div>

        {/* Base Map Layers */}
        {baseLayer === 'satellite' ? (
          <>
            <TileLayer
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              attribution="&copy; Esri"
            />
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png"
              subdomains="abcd"
            />
          </>
        ) : baseLayer === 'terrain' ? (
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            attribution="&copy; CartoDB"
            subdomains="abcd"
          />
        ) : (
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
            attribution="&copy; Esri"
          />
        )}

          {/* Dynamic GeoJSON Boundary for Pakistan */}
          {geoData && (
            <GeoJSON 
              data={geoData} 
              style={{ color: '#111827', weight: 4, fillOpacity: 0, opacity: 0.8 }} 
            />
          )}

          {/* Pakistan Water Bodies (Rivers, Canals, Dams) */}
          {waterBodiesData && (
            <GeoJSON 
              key={isEmergency ? 'red' : 'blue'}
              data={waterBodiesData} 
              style={(feature) => ({
                color: isEmergency ? '#ef4444' : '#2563eb', // Vibrant blue normally, bright red in emergency
                weight: feature.properties.type === 'river' ? 4 : 2, 
                opacity: 1.0 
              })}
              pointToLayer={(feature, latlng) => {
                const label = feature.properties.name.replace(/(River|Dam|Barrage)/g, '').trim();
                const isDanger = feature.properties.inflow && feature.properties.inflow >= feature.properties.capacity;
                
                let svgIcon = '';
                if (feature.properties.type === 'dam') {
                  const color = isDanger ? '#dc2626' : '#f97316'; // Red if danger, Orange normally
                  const stroke = isDanger ? '#7f1d1d' : '#c2410c';
                  svgIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="${color}" stroke="${stroke}" stroke-width="2" style="filter: drop-shadow(0px 3px 3px rgba(0,0,0,0.4));"><polygon points="12 2 22 20 2 20" /></svg>`;
                } else if (feature.properties.type === 'barrage') {
                  const color = isDanger ? '#dc2626' : '#22c55e'; // Red if danger, Green normally
                  const stroke = isDanger ? '#7f1d1d' : '#166534';
                  svgIcon = `<svg width="18" height="18" viewBox="0 0 24 24" fill="${color}" stroke="${stroke}" stroke-width="2" style="filter: drop-shadow(0px 3px 3px rgba(0,0,0,0.4));"><circle cx="12" cy="12" r="10" /></svg>`;
                }
                
                if (svgIcon) {
                  const html = `<div style="display:flex; flex-direction:column; align-items:center;">
                                  ${svgIcon}
                                  <span class="mt-0.5 text-[10px] font-black ${isDanger ? 'text-red-600' : 'text-gray-800'} drop-shadow-md" style="text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;">${label}</span>
                                  ${isDanger ? '<span class="w-2 h-2 rounded-full bg-red-600 animate-pulse mt-0.5 border border-white shadow-sm"></span>' : ''}
                                </div>`;
                  return L.marker(latlng, { icon: L.divIcon({ html, className: '', iconSize: [100, 40], iconAnchor: [50, 15] }) });
                }
                return L.marker(latlng);
              }}
              onEachFeature={(feature, layer) => {
                if (feature.properties && feature.properties.name) {
                  let tooltip = `<strong class="text-[11px] text-gray-800">${feature.properties.name}</strong>`;
                  if (feature.properties.inflow !== undefined) {
                    const dangerText = feature.properties.inflow >= feature.properties.capacity ? '<br/><span class="text-red-500 font-bold text-[10px]">⚠️ OVERFLOW DANGER</span>' : '';
                    tooltip += `<div class="mt-1 border-t border-gray-200 pt-1">
                                  <div class="text-[10px] text-gray-600">Inflow: <span class="font-bold text-blue-600">${feature.properties.inflow.toLocaleString()} Cs</span></div>
                                  <div class="text-[10px] text-gray-600">Outflow: <span class="font-bold text-orange-500">${feature.properties.outflow.toLocaleString()} Cs</span></div>
                                  <div class="text-[10px] text-gray-600 mt-1">Capacity: <span class="font-bold text-gray-800">${feature.properties.capacity.toLocaleString()} Cs</span></div>
                                  ${dangerText}
                                </div>`;
                  }
                  layer.bindTooltip(tooltip, { 
                    permanent: false, 
                    direction: 'top',
                    className: 'bg-white/95 border border-gray-200 rounded-xl shadow-2xl px-3 py-2'
                  });
                }
              }}
            />
          )}

          {/* Animated Live Wind Flow Particle System */}
          {windEnabled && <VelocityLayer baseLayer={baseLayer} />}

          {/* Live Weather Overlays from OpenWeatherMap (Multi-Layer Rendering) */}
          {activeLayers.map(layerId => (
            <TileLayer
              key={getOwmLayerId(layerId)}
              url={`https://tile.openweathermap.org/map/${getOwmLayerId(layerId)}/{z}/{x}/{y}.png?appid=d1ccaf30c2e39ba82506f4ed9bdb50f2`}
              opacity={0.8} // Lowered opacity so multiple layers blend better
              zIndex={500}
            />
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
