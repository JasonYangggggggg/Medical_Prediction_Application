import React, { useState } from "react";
import Cookies from "js-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faMapMarkerAlt } from "@fortawesome/free-solid-svg-icons";

const WelcomeModal = ({ isOpen, onClose, onSubmit, autoLocation = null }) => {
  const [zip, setZip] = useState(autoLocation?.zip || "");
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);

  // --------------------------
  // SUBMIT HANDLER
  // --------------------------
  const handleSubmit = async () => {
    // If we have auto-location, use it
    if (autoLocation && autoLocation.zip) {
      localStorage.setItem("userZip", autoLocation.zip);
      localStorage.setItem("userLatitude", autoLocation.lat);
      localStorage.setItem("userLongitude", autoLocation.lon);
      onSubmit(autoLocation.zip);
      onClose();
      return;
    }

    // Otherwise, validate manual input
    if (zip.trim().length !== 3 || !address.trim()) {
      alert("Please enter a valid ZIP code (3 characters) and address.");
      return;
    }

    try {
      // 1) Fetch coordinates via OpenStreetMap API
      const response = await fetch(
        `http://26.81.189.101:8080/search?q=${encodeURIComponent(address)}&format=json&addressdetails=1&limit=1`
      );
      const data = await response.json();

      // 2) If we get a result back, save lat & lon in localStorage
      if (data.length > 0) {
        const { lat, lon } = data[0];
        localStorage.setItem("userLatitude", lat);
        localStorage.setItem("userLongitude", lon);

        // 3) Also save ZIP to cookies AND to localStorage
        Cookies.set("userZip", zip, { expires: 7 });
        localStorage.setItem("userZip", zip); // <-- Save ZIP in localStorage too

        // 4) Notify parent component
        onSubmit(zip);
        onClose();
      } else {
        alert("Unable to find coordinates for the given address.");
      }
    } catch (error) {
      console.error("Error fetching latitude and longitude:", error);
      alert("An error occurred while processing your address.");
    }
  };

  // --------------------------
  // FIND MY LOCATION HANDLER
  // --------------------------
  const handleFindMyLocation = async () => {
    try {
      setLoading(true);

      // 1) Use ip-api.com to get user's approximate lat/lon
      const response = await fetch("http://ip-api.com/json/");
      const data = await response.json();

      if (data.lat && data.lon) {
        // 2) Save lat/lon to localStorage
        localStorage.setItem("userLatitude", data.lat);
        localStorage.setItem("userLongitude", data.lon);

        // 3) If IP-API provides a ZIP, truncate it to 3 chars
        const truncatedZip = data.zip?.slice(0, 3) || "";

        // 4) If ZIP is available, store it in cookies & localStorage
        if (truncatedZip) {
          Cookies.set("userZip", truncatedZip, { expires: 7 });
          localStorage.setItem("userZip", truncatedZip); // <-- Save ZIP in localStorage too
        }

        // 5) Close modal
        onClose();
      } else {
        alert("Unable to find your location. Please try again.");
      }
    } catch (error) {
      console.error("Error fetching IP-based location:", error);
      alert("An error occurred while fetching your location.");
    } finally {
      setLoading(false);
    }
  };

  // --------------------------
  // RENDER NOTHING IF NOT OPEN
  // --------------------------
  if (!isOpen) return null;

  // --------------------------
  // MODAL UI
  // --------------------------
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-[9999]">
      <div className="bg-white p-6 rounded-lg shadow-lg w-[90%] max-w-md">
        <h2 className="text-xl font-bold mb-4 text-center text-yellow-500">
          Welcome to Bargain Bee
        </h2>
        
        {autoLocation ? (
          <div className="mb-6">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-green-600">📍</span>
                <span className="font-semibold text-green-800">Location Detected!</span>
              </div>
              <p className="text-green-700 text-sm">
                {autoLocation.city}, {autoLocation.region}<br/>
                ZIP: {autoLocation.zip}
              </p>
            </div>
            <p className="text-black text-center mb-4">
              We've automatically detected your location. Click continue to start shopping!
            </p>
          </div>
        ) : (
          <div>
            <p className="text-black text-center mb-6">
              Please enter your address and ZIP code to get started, or use "Find My Location".
            </p>

            {/* Address Input */}
            <div className="mb-4">
              <label htmlFor="address" className="block text-sm font-medium text-black mb-1">
                Enter Address
              </label>
              <input
                type="text"
                id="address"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
                placeholder="Enter your full address"
              />
            </div>

            {/* ZIP Code Input */}
            <div className="mb-4">
              <label htmlFor="zip" className="block text-sm font-medium text-black mb-1">
                Enter ZIP (3 characters)
              </label>
              <input
                type="text"
                id="zip"
                value={zip}
                onChange={(e) => setZip(e.target.value)}
                maxLength={3}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
                placeholder="Enter ZIP Code"
              />
            </div>

            {/* Find My Location Button */}
            <div className="mb-4">
              <button
                onClick={handleFindMyLocation}
                className={`flex items-center justify-center w-full bg-yellow-400 hover:bg-yellow-500 text-white px-4 py-2 rounded-lg ${
                  loading ? "cursor-wait" : ""
                }`}
                disabled={loading}
              >
                {loading ? (
                  "Finding..."
                ) : (
                  <>
                    <FontAwesomeIcon icon={faMapMarkerAlt} className="mr-2" />
                    Find My Location!
                  </>
                )}
              </button>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-4">
              <button
                onClick={onClose}
                className="bg-gray-200 hover:bg-gray-300 text-black px-4 py-2 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                className="bg-yellow-400 hover:bg-yellow-500 text-white px-4 py-2 rounded-lg"
              >
                Submit
              </button>
            </div>
          </div>
        )}

        {/* Action Buttons for Auto-detected Location */}
        {autoLocation && (
          <div className="flex justify-end gap-4">
            <button
              onClick={onClose}
              className="bg-gray-200 hover:bg-gray-300 text-black px-4 py-2 rounded-lg"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              className="bg-yellow-400 hover:bg-yellow-500 text-white px-4 py-2 rounded-lg"
            >
              Continue
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default WelcomeModal;
