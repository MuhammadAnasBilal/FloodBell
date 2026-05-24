import { useEffect, useState } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-velocity';

export default function VelocityLayer({ baseLayer = 'satellite', displayValues = true, displayOptions = {} }) {
  const map = useMap();
  const [windData, setWindData] = useState(null);

  useEffect(() => {
    fetch('/wind-global.json')
      .then(res => res.json())
      .then(data => setWindData(data));
  }, []);

  useEffect(() => {
    if (!map || !windData) return;

    const velocityLayer = L.velocityLayer({
      displayValues: displayValues,
      displayOptions: {
        velocityType: 'Global Wind',
        position: 'bottomleft',
        emptyString: 'No wind data',
        angleConvention: 'bearingCW',
        displayPosition: 'bottomleft',
        displayEmptyString: 'No wind data',
        speedUnit: 'm/s',
        ...displayOptions
      },
      data: windData,
      maxVelocity: 15,
      velocityScale: 0.01,
      particleAge: 90,
      lineWidth: 3,
      particleMultiplier: 1 / 500,
      interactive: false, // Prevents stopping on mouse hover
      colorScale: baseLayer === 'terrain' || baseLayer === 'street' 
        ? ["rgba(59,130,246,0.6)", "rgba(37,99,235,0.8)", "rgba(29,78,216,1.0)", "rgba(30,64,175,1.0)"] // Darker blue for light maps
        : ["rgba(255,255,255,0.7)", "rgba(255,255,255,0.9)", "rgba(0,255,255,1.0)", "rgba(0,191,255,1.0)"] // Bright for satellite
    });

    velocityLayer.addTo(map);

    return () => {
      map.removeLayer(velocityLayer);
    };
  }, [map, windData, displayValues, displayOptions]);

  return null;
}
