import React, { useState, useEffect } from "react";
import Card from "./Card";
import { motion, AnimatePresence } from "framer-motion";

const Sidebar = ({ products, onCardClick, userList, onAddToList }) => {
  const [selectedProductName, setSelectedProductName] = useState("All");

  // Extract unique product titles, add "All" option at the beginning
  const uniqueProductTitles = ["All", ...new Set(products.map((product) => product.title))].sort((a, b) => {
    if (a === "All") return -1; // Keep "All" at the top
    if (b === "All") return 1;
    return a.localeCompare(b);
  });

  // Reset to "All" when products change (i.e., after a new search)
  useEffect(() => {
    setSelectedProductName("All");
  }, [products]);

  const handleProductChange = (e) => {
    setSelectedProductName(e.target.value);
  };

  // Show all products when "All" is selected, otherwise filter
  const filteredProducts = selectedProductName === "All" 
    ? products 
    : products.filter((product) => product.title === selectedProductName);

  return (
    <div className="w-[22rem] bg-gray-100 shadow-lg overflow-y-auto">
      <div className="p-4 pt-16">
        {/* Dropdown Filter */}
        <div className="mb-3 mx-auto sm:w-[90%] md:w-[80%] lg:w-full">
          <label htmlFor="product-filter" className="block text-sm font-medium text-gray-700">
            Filter by Product Name
          </label>
          <select
            id="product-filter"
            value={selectedProductName}
            onChange={handleProductChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm focus:ring-yellow-500 focus:border-yellow-500 sm:text-sm"
          >
            {uniqueProductTitles.map((title, index) => (
              <option key={index} value={title}>
                {title}
              </option>
            ))}
          </select>
        </div>

        {/* Cards */}
        <AnimatePresence>
          <div className="flex flex-wrap gap-4 justify-center">
            {filteredProducts.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.5 }}
                className="
                  flex flex-col items-center justify-center text-center bg-white 
                  shadow-md rounded-lg p-6 border border-gray-200 
                  w-full sm:w-[90%] md:w-[80%] lg:w-full
                "
              >
                {/* Floating Icon */}
                <motion.div
                  className="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center mb-4"
                  initial={{ y: 0 }}
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-8 w-8 text-yellow-500"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v3a1 1 0 00.293.707l2 2a1 1 0 001.414-1.414L11 9.586V6z"
                      clipRule="evenodd"
                    />
                  </svg>
                </motion.div>
                <h2 className="text-lg font-medium text-gray-700">
                  No Results Found
                </h2>
                <p className="text-sm text-gray-500 mt-2">
                  Try searching for a product to get started!
                </p>
              </motion.div>
            ) : (
              filteredProducts.map((product, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 40 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 40 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  className="
                    transform scale-100 
                    sm:scale-95 md:scale-90 lg:scale-100 
                    w-full sm:w-[90%] md:w-[80%] lg:w-full 
                    mx-auto
                  "
                >
                  <Card
                    product={product}
                    userList={userList}
                    onAddToList={onAddToList}
                    onCardClick={onCardClick}
                  />
                </motion.div>
              ))
            )}
          </div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Sidebar;
