import React, { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { motion } from "framer-motion";

const MapComponent = ({ storeLocations, center, zipCode }) => {
  const mapRef = useRef(null);
  const markersRef = useRef([]); // Store references to all markers
  const userMarkerRef = useRef(null); // Reference to the user marker

  // Helper function to compare lat/lng with a small tolerance
  const isCoordinatesMatch = (lat1, lon1, lat2, lon2, tolerance = 0.0001) => {
    return Math.abs(lat1 - lat2) < tolerance && Math.abs(lon1 - lon2) < tolerance;
  };

  useEffect(() => {
    // Initialize the map on mount
    mapRef.current = L.map("map", {
      zoomControl: false,
      center: [center.lat, center.lng],
      zoom: 13, // Zoom level
    });

    // Add OpenStreetMap tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(mapRef.current);

    return () => {
      mapRef.current.remove();
    };
  }, []);

  useEffect(() => {
    // Clear existing markers when storeLocations update
    markersRef.current.forEach((marker) => {
      mapRef.current.removeLayer(marker);
    });
    markersRef.current = [];

    // Add store location markers
    storeLocations.forEach((location) => {
      const marker = L.marker([location.latitude, location.longitude], {
        icon: L.divIcon({
          className: "default-icon",
          html: `<div style="text-align: center; width: 32px; height: 32px;">
                    <i class="fas fa-map-marker-alt" 
                       style="color: black; font-size: 32px; text-shadow: -2px -2px 0 white, 2px -2px 0 white, -2px 2px 0 white, 2px 2px 0 white;"></i>
                 </div>`,
        }),
      })
        .addTo(mapRef.current)
        .bindPopup(
          `<b>${location.title}</b><br>${location.address}<br>${location.distance}`
        );

      markersRef.current.push(marker);
    });
  }, [storeLocations]);

  useEffect(() => {
    // Highlight the marker that matches the current center
    markersRef.current.forEach((marker) => {
      const { lat, lng } = marker.getLatLng();
      const isCenter = isCoordinatesMatch(lat, lng, center.lat, center.lng);

      marker.setIcon(
        L.divIcon({
          className: isCenter ? "highlighted-icon" : "default-icon",
          html: `<div style="text-align: center; width: 32px; height: 32px;">
                    <i class="fas fa-map-marker-alt" 
                       style="color: ${
                         isCenter ? "red" : "black"
                       }; font-size: 32px; text-shadow: ${
            isCenter
              ? "-2px -2px 0 white, 2px -2px 0 white, -2px 2px 0 white, 2px 2px 0 white"
              : "-2px -2px 0 white, 2px -2px 0 white, -2px 2px 0 white, 2px 2px 0 white"
          };"></i>
                 </div>`,
        })
      );
    });

    // Fly to the center with animation
    mapRef.current.flyTo([center.lat, center.lng], 13, { duration: 1 });
  }, [center]);

  useEffect(() => {
    // Add a marker for the user's saved location
    const savedLatitude = localStorage.getItem("userLatitude");
    const savedLongitude = localStorage.getItem("userLongitude");

    if (savedLatitude && savedLongitude) {
      if (userMarkerRef.current) {
        mapRef.current.removeLayer(userMarkerRef.current); // Remove previous marker if it exists
      }

      userMarkerRef.current = L.marker([parseFloat(savedLatitude), parseFloat(savedLongitude)], {
        icon: L.divIcon({
          className: "custom-user-marker",
          html: `<div style="text-align: center; color: #28a745; font-size: 2.5rem;">
                   <i class="fas fa-user-circle"></i>
                 </div>`,
        }),
      }).addTo(mapRef.current);
    }
  }, []);

  useEffect(() => {
    // Fetch and display the marker for the ZIP code location
    const fetchZipLocation = async () => {
      if (!zipCode) return;

      try {
        const response = await fetch(
          `http://127.0.0.1:8080/search?q=${zipCode}&format=json&addressdetails=1&limit=1`
        );

        if (response.ok) {
          const data = await response.json();
          if (data.length > 0) {
            const { lat, lon } = data[0];

            // Add a custom marker for the ZIP code
            L.marker([lat, lon], {
              icon: L.divIcon({
                className: "custom-zip-marker",
                html: `<i class="fas fa-map-pin fa-3x text-red-500"></i>`,
              }),
            })
              .addTo(mapRef.current)
              .bindPopup(`Saved ZIP Code: ${zipCode}`);
          }
        }
      } catch (error) {
        console.error("Error fetching ZIP code location:", error);
      }
    };

    fetchZipLocation();
  }, [zipCode]);

  return (
    <motion.div
      id="map"
      className="fixed top-0 left-[80px] w-[calc(100%-80px)] h-screen z-0"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    />
  );
};

export default MapComponent;
