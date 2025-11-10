import React from "react";
import Cookies from "js-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faSearch, faMapMarkerAlt } from "@fortawesome/free-solid-svg-icons";
import { useNavigate } from "react-router-dom";

const SearchBar = ({ productName, setProductName, scrapeFirstStores, scraping, openModal, inputRef }) => {
  const navigate = useNavigate();

  // Retrieve ZIP code from cookies
  const userZip = Cookies.get("userZip") || "Enter ZIP";

  const handleSearchBarClick = () => {
    if (inputRef?.current) {
      inputRef.current.focus(); // Ensure the input is focused
    }
    navigate("/map"); // Navigate to the map page
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !scraping) {
      scrapeFirstStores(); // Trigger the search when Enter is pressed
    }
  };

  return (
    <div
      className="relative flex items-center bg-white shadow-lg rounded-full px-2 py-1 sm:px-3 sm:py-1 w-[90%] max-w-[320px] sm:max-w-lg mx-auto z-[9999] overflow-hidden"
      onClick={handleSearchBarClick} // Navigate to the map page when the search bar is clicked
    >
      {/* Search Icon */}
      <div
        className="flex items-center justify-center w-8 h-8 sm:w-10 sm:h-10 bg-yellow-300 rounded-full cursor-pointer aspect-square shrink-0 -ml-[4px]" // Move 1px to the left
        onClick={(e) => {
          e.stopPropagation(); // Prevent triggering navigation when clicking the search icon
          if (!scraping) scrapeFirstStores();
        }}
      >
        {!scraping ? (
          <FontAwesomeIcon icon={faSearch} className="text-black text-xs sm:text-sm" />
        ) : (
          <div className="w-4 h-4 sm:w-5 sm:h-5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
        )}
      </div>

      {/* Input Field */}
      <input
        ref={inputRef} // Attach ref to input field
        type="text"
        placeholder="Search Product"
        value={productName}
        onChange={(e) => setProductName(e.target.value)}
        onKeyDown={handleKeyDown} // Listen for the Enter key
        className="flex-grow text-xs sm:text-sm text-gray-700 bg-transparent focus:outline-none placeholder-gray-400 ml-3 min-w-0"
      />

      
    </div>
  );
};

export default SearchBar;
