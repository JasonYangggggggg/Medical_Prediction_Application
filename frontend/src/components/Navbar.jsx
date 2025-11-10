import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faB, faHome, faMap, faList, faSignOutAlt,faDownload } from "@fortawesome/free-solid-svg-icons";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();

  const handleLogout = () => {
    // Clear authentication details from localStorage
    localStorage.removeItem("token");
    localStorage.removeItem("isGuest");
    // Redirect to the login page
    window.location.href = "/";
  };

  return (
    <nav className="fixed top-0 left-0 h-full bg-white shadow-md flex flex-col items-center px-4 py-6 z-50 w-20">
      {/* Logo */}
      <div className="flex items-center justify-center text-yellow-500 text-3xl font-bold mb-10">
         <Link to="/" className="flex flex-col items-center justify-center">
        <FontAwesomeIcon icon={faB} />
        </Link>
      </div>

      {/* Navigation Links */}
      <ul className="flex flex-col space-y-6 text-gray-700 w-full items-center">
        <li
          className={`hover:text-yellow-500 cursor-pointer flex flex-col items-center justify-center ${
            location.pathname === "/" ? "text-yellow-500 font-bold" : ""
          }`}
        >
          <Link to="/" className="flex flex-col items-center justify-center">
            <FontAwesomeIcon icon={faHome} className="text-xl" />
            <span className="text-xs mt-1">Home</span>
          </Link>
        </li>
        <li
          className={`hover:text-yellow-500 cursor-pointer flex flex-col items-center justify-center ${
            location.pathname === "/download" ? "text-yellow-500 font-bold" : ""
          }`}
        >
          <Link to="/download" className="flex flex-col items-center justify-center">
            <FontAwesomeIcon icon={faDownload} className="text-xl" />
            <span className="text-xs mt-1">Download</span>
          </Link>
        </li>
        <li
          className={`hover:text-yellow-500 cursor-pointer flex flex-col items-center justify-center ${
            location.pathname === "/map" ? "text-yellow-500 font-bold" : ""
          }`}
        >
          <Link to="/map" className="flex flex-col items-center justify-center">
            <FontAwesomeIcon icon={faMap} className="text-xl" />
            <span className="text-xs mt-1">Map</span>
          </Link>
        </li>
        <li
          className={`hover:text-yellow-500 cursor-pointer flex flex-col items-center justify-center ${
            location.pathname === "/my-list" ? "text-yellow-500 font-bold" : ""
          }`}
        >
          <Link to="/my-list" className="flex flex-col items-center justify-center">
            <FontAwesomeIcon icon={faList} className="text-xl" />
            <span className="text-xs mt-1">List</span>
          </Link>
        </li>
        <li
          className="hover:text-yellow-500 cursor-pointer flex flex-col items-center justify-center"
          onClick={handleLogout}
        >
          <FontAwesomeIcon icon={faSignOutAlt} className="text-xl" />
          <span className="text-xs mt-1">Log Out</span>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
