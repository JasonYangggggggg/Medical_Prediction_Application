import React, { useEffect, useState } from "react";
import Cookies from "js-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faMapMarkerAlt } from "@fortawesome/free-solid-svg-icons";
import { useNavigate } from "react-router-dom";
import { scraperApi } from "../utils/api";

const SoftwareInstall = ({ isOpen, onClose, onSubmit }) => {
  const [zip, setZip] = useState("");
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    // Check if the REST server is running
    scraperApi.healthCheck()
      .then((isHealthy) => {
        if (isHealthy) setShowModal(false); // Hide modal if server is running
      })
      .catch(() => {
        setShowModal(true); // Show modal if server is not running
      });
  }, []);

  // --------------------------
  // SUBMIT HANDLER
  // --------------------------
  const handleSubmit = () => {
    navigate("/download");
  };

  // --------------------------
  // FIND MY LOCATION HANDLER
  // --------------------------
  const handleFindMyLocation = async () => {
   
  };

  // --------------------------
  // RENDER NOTHING IF NOT OPEN
  // --------------------------
  if (!isOpen || !showModal) return null;

  // --------------------------
  // MODAL UI
  // --------------------------
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-60 z-[9999] backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl w-[90%] max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-yellow-400 to-yellow-500 px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <div className="w-6 h-6 bg-white rounded-full flex items-center justify-center">
                <span className="text-yellow-500 text-sm font-bold">B</span>
              </div>
            </div>
            <h2 className="text-2xl font-semibold text-white">
              Bargain Bee
            </h2>
          </div>
        </div>

        {/* Content */}
        <div className="px-8 py-8">
          <div className="text-center mb-8">
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Complete Your Setup
            </h3>
            <p className="text-gray-600 leading-relaxed mb-4">
              Install our desktop companion app to unlock Bargain Bee’s full potential—it will automatically perform product searches whenever you click Search.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-blue-800 text-sm font-medium">
                � This application requires our helper software to be installed on your device to function properly. The app will not work at all without this required component.
              </p>
            </div>
          </div>

          {/* Features */}
          <div className="space-y-3 mb-8">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 bg-yellow-100 rounded-full flex items-center justify-center flex-shrink-0">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              </div>
              <span className="text-gray-700 text-sm">One-click product search via deep-links</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 bg-yellow-100 rounded-full flex items-center justify-center flex-shrink-0">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              </div>
              <span className="text-gray-700 text-sm">Automatic store location mapping</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
                <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
              </div>
              <span className="text-gray-500 text-sm flex items-center gap-2">
                AI shopping assistant & chat 
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full font-medium">Coming Soon</span>
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col-reverse sm:flex-row gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 text-gray-600 hover:text-gray-800 hover:bg-gray-50 rounded-xl transition-all duration-200 font-medium border border-gray-200"
            >
              Maybe Later
            </button>
            <button
              onClick={handleSubmit}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-yellow-400 to-yellow-500 hover:from-yellow-500 hover:to-yellow-600 text-white rounded-xl transition-all duration-200 font-semibold shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              Install Helper App
            </button>
          </div>

          {/* Trust indicators */}
          <div className="mt-6 pt-6 border-t border-gray-100">
            <div className="flex items-center justify-center gap-6 text-xs text-gray-500">
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                Secure & Private
              </span>
              <span className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
                Free Forever
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SoftwareInstall;
