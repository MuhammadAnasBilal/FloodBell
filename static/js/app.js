/* ============================================================
   FloodGuard Pakistan V3 - Ventusky Style JS
   ============================================================ */

   const OWM_KEY = 'd1ccaf30c2e39ba82506f4ed9bdb50f2';
   let map, userMarker, redZoneCircle;
   let userLat = null, userLng = null;
   let overlays = {};
   let baseLayers = {};
   let currentUser = null;
   
   document.addEventListener('DOMContentLoaded', () => {
       setupAuthMock(); // Replace with real Firebase later
       initMap();
       setupSearch();
       fetchIntelligence();
   });
   
   // --- AUTHENTICATION MOCK ---
   function setupAuthMock() {
       const modal = document.getElementById('authModal');
       const btn = document.getElementById('btnGoogleSignIn');
       
       btn.addEventListener('click', () => {
           // Show loading state
           btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Authenticating with Google...';
           btn.disabled = true;
           
           // Simulate network delay for realistic process
           setTimeout(() => {
               currentUser = {
                   displayName: "Test User",
                   email: "test@example.com",
                   photoURL: "https://ui-avatars.com/api/?name=Test+User&background=3b82f6&color=fff"
               };
               
               document.getElementById('userName').textContent = currentUser.displayName;
               const avatar = document.getElementById('userAvatar');
               avatar.src = currentUser.photoURL;
               avatar.classList.remove('hidden');
               
               modal.classList.remove('active');
               document.getElementById('mainNav').classList.remove('hidden');
               document.getElementById('appContent').classList.remove('hidden');
               document.getElementById('chatFabContainer').classList.remove('hidden');
               
               // Fix Map rendering bug by invalidating size after container becomes visible
               setTimeout(() => {
                   map.invalidateSize();
                   requestUserLocation();
               }, 300);
           }, 1500);
       });
   }
   
   // --- MAP INIT ---
   function initMap() {
       const pakBounds = L.latLngBounds([23.5, 60.5], [37.5, 78.0]);
       
       map = L.map('map', {
           center: [30.3753, 69.3451], // Default center Pakistan
           zoom: 6,
           maxBounds: pakBounds,
           maxBoundsViscosity: 1.0,
           minZoom: 5,
           zoomControl: false // Hide default zoom for cleaner look
       });
       
       L.control.zoom({ position: 'bottomleft' }).addTo(map);
   
       baseLayers['satellite'] = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {attribution: 'Esri'}).addTo(map);
       baseLayers['dark'] = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {attribution: 'CartoDB'});
   
       overlays['precipitation_new'] = L.tileLayer(`https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png?appid=${OWM_KEY}`, {opacity: 0.8});
       overlays['wind_new'] = L.tileLayer(`https://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?appid=${OWM_KEY}`, {opacity: 0.6});
       overlays['temp_new'] = L.tileLayer(`https://tile.openweathermap.org/map/temp_new/{z}/{x}/{y}.png?appid=${OWM_KEY}`, {opacity: 0.5});
   
       map.on('click', (e) => onLocationSelected(e.latlng.lat, e.latlng.lng, 'Selected Point'));
   }
   
   function setBaseLayer(name, btn) {
       document.querySelectorAll('.layer-btn').forEach(b => { if(!b.classList.contains('gps-btn')) b.classList.remove('active'); });
       btn.classList.add('active');
       
       Object.values(baseLayers).forEach(layer => map.removeLayer(layer));
       baseLayers[name].addTo(map);
   }
   
   function toggleOverlay(name, isChecked) {
       if(isChecked) {
           overlays[name].addTo(map);
       } else {
           map.removeLayer(overlays[name]);
       }
   }
   
   function requestUserLocation() {
       if (navigator.geolocation) {
           navigator.geolocation.getCurrentPosition(
               pos => {
                   onLocationSelected(pos.coords.latitude, pos.coords.longitude, 'My Location');
                   map.setView([pos.coords.latitude, pos.coords.longitude], 12); // Zoom to 10km view
               },
               err => console.log('Geolocation denied or failed.'),
               {enableHighAccuracy: true}
           );
       }
   }
   
   function onLocationSelected(lat, lng, name) {
       userLat = lat; userLng = lng;
       
       if (userMarker) map.removeLayer(userMarker);
       const icon = L.divIcon({ className: 'gps-pulse', iconSize: [20, 20], iconAnchor: [10, 10] });
       userMarker = L.marker([lat, lng], {icon: icon}).addTo(map);
       
       analyzeRisk(lat, lng);
   }
   
   // --- PREDICTION ENGINE ---
   function analyzeRisk(lat, lng) {
       document.getElementById('riskBadge').textContent = "ANALYZING...";
       document.getElementById('riskBadge').style.borderColor = "#94a3b8";
       document.getElementById('riskBadge').style.color = "#94a3b8";
       
       fetch('/api/predict/location', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify({lat, lng})
       })
       .then(r => r.json())
       .then(data => {
           if(data.success) {
               updateRiskUI(data, lat, lng);
           }
       });
   }
   
   function updateRiskUI(data, lat, lng) {
       const risk = data.risk_level;
       const prob = data.flood_probability;
       const conf = data.confidence;
       const colors = {low: 'var(--safe)', medium: 'var(--medium)', high: 'var(--high)', critical: 'var(--critical)'};
       
       const badge = document.getElementById('riskBadge');
       badge.textContent = risk.toUpperCase();
       badge.style.borderColor = colors[risk];
       badge.style.color = colors[risk];
       
       document.getElementById('riskProb').textContent = `${(prob*100).toFixed(1)}%`;
       document.getElementById('riskConf').textContent = `${conf}%`;
       
       const driverList = document.getElementById('driverList');
       driverList.innerHTML = data.drivers.map(d => {
           const hasArrow = d.includes('↑') || d.includes('↓');
           const text = hasArrow ? d.substring(2) : d;
           const icon = d.includes('↑') ? '<i class="fas fa-arrow-trend-up" style="color:var(--critical)"></i>' : 
                        d.includes('↓') ? '<i class="fas fa-arrow-trend-down" style="color:var(--safe)"></i>' : 
                        '<i class="fas fa-minus" style="color:var(--medium)"></i>';
           return `<li>${icon} ${text}</li>`;
       }).join('');
       
       handleRedZone(risk, lat, lng);
   }
   
   function handleRedZone(risk, lat, lng) {
       const evacPanel = document.getElementById('evacPanel');
       
       if(redZoneCircle) map.removeLayer(redZoneCircle);
       
       if(risk === 'critical' || risk === 'high') {
           // Draw 50km Red Zone
           redZoneCircle = L.circle([lat, lng], {
               color: 'red',
               fillColor: '#f03',
               fillOpacity: 0.3,
               radius: 50000 // 50km
           }).addTo(map);
           
           evacPanel.classList.remove('hidden');
           
           // Fetch nearest shelter logic
           fetch(`/api/data/irsa`) // In reality, we'd have a shelter endpoint. Using placeholder.
           document.getElementById('evacTarget').innerHTML = "Routing to <b>Safe Zone (Outside 50km)</b>...";
           
           // Google Maps out of red zone (adding 0.5 degrees lat to simulate moving 50km North)
           const safeLat = lat + 0.5; 
           document.getElementById('directionsBtn').href = `https://www.google.com/maps/dir/?api=1&origin=${lat},${lng}&destination=${safeLat},${lng}`;
       } else {
           evacPanel.classList.add('hidden');
       }
   }
   
   function sendMockAlert() {
       if(!currentUser) return;
       fetch('/api/alerts/send', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify({
               user_email: currentUser.email,
               risk_level: document.getElementById('riskBadge').textContent,
               location: `${userLat}, ${userLng}`,
               distance_km: 52
           })
       }).then(r => r.json()).then(data => alert(data.message));
   }
   
   // --- SEARCH ---
   function setupSearch() {
       const input = document.getElementById('searchInput');
       input.addEventListener('keypress', (e) => {
           if(e.key === 'Enter' && input.value) {
               const query = input.value;
               input.value = "Searching...";
               fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${query}&countrycodes=pk`)
               .then(r => r.json())
               .then(data => {
                   input.value = "";
                   if (data && data.length > 0) {
                       const lat = parseFloat(data[0].lat);
                       const lon = parseFloat(data[0].lon);
                       onLocationSelected(lat, lon, data[0].display_name);
                       map.setView([lat, lon], 12);
                   } else {
                       alert("Location not found in Pakistan.");
                   }
               })
               .catch(err => {
                   input.value = "";
                   alert("Search API Error.");
               });
           }
       });
   }
   
   // --- DATA INTELLIGENCE ---
   function fetchIntelligence() {
       fetch('/api/data/irsa').then(r=>r.json()).then(data => {
           document.getElementById('irsaRawData').textContent = JSON.stringify(data, null, 2);
       });
   }
   
   // --- TABS & CHATBOT ---
   function switchTab(tab) {
       document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
       document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
       document.getElementById('tab-'+tab).classList.add('active');
       document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
       if(tab === 'dashboard' && map) setTimeout(()=>map.invalidateSize(), 100);
   }
   
   function toggleChat() {
       const widget = document.getElementById('chatWidget');
       widget.classList.toggle('hidden');
   }
   
   function sendChat() {
       const input = document.getElementById('chatInput');
       const msg = input.value.trim();
       if(!msg) return;
       
       addChatBubble(msg, 'user');
       input.value = '';
       const id = addChatBubble("Grok is analyzing...", 'bot');
       
       fetch('/api/chat/', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify({message: msg, lat: userLat || 0, lng: userLng || 0})
       }).then(r=>r.json()).then(data => {
           document.getElementById(id).remove();
           if(data.success) addChatBubble(data.response, 'bot');
           else addChatBubble("Failed to reach intelligence server.", 'bot');
       });
   }
   
   function addChatBubble(text, type) {
       const container = document.getElementById('chatMessages');
       const id = 'msg-' + Date.now();
       container.insertAdjacentHTML('beforeend', `<div class="msg ${type}" id="${id}">${text}</div>`);
       container.scrollTop = container.scrollHeight;
       return id;
   }
