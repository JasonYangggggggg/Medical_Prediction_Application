import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCirclePlus, faCheckCircle } from "@fortawesome/free-solid-svg-icons";

const Card = ({ product, userList, onAddToList, onCardClick }) => {
  const [isInList, setIsInList] = useState(false);

  useEffect(() => {
    const itemExists = userList.some(
      (item) =>
        item.title === product.title &&
        item.store_address === product.store_address
    );
    setIsInList(itemExists);
  }, [userList, product.title, product.store_address]);

  const handleAddToList = (e) => {
    e.stopPropagation(); // Prevent triggering `handleCardClick`
    if (!isInList) {
      onAddToList(product);
      setIsInList(true);
    }
  };

  const handleCardClick = () => {
    if (onCardClick) {
      onCardClick(product.store_latitude, product.store_longitude);
    }
  };

  return (
    <motion.div
      className="
        relative
        /* In mobile mode, set width to full viewport minus navbar width */
        w-[calc(100vw-5rem)]
        p-3
        bg-white
        rounded-[12px]
        border border-gray-300
        flex gap-3
        shadow
        cursor-pointer
        font-sans
        text-xs
        /* On larger screens, use a fixed max width */
        sm:max-w-[19rem] sm:p-3 sm:mt-4
      "
      onClick={handleCardClick}
      whileHover={{ scale: 1.02, translateY: -5 }}
      whileTap={{ scale: 0.98 }}
      transition={{ duration: 0.3 }}
    >
      {/* Image Section */}
      <div className="flex flex-col items-center">
        {product.image && (
          <img
            src={product.image}
            alt={product.title}
            className="
              object-contain
              rounded-[10px]
              border border-gray-300
              w-[90px] h-[110px]
              sm:w-[100px] sm:h-[120px]
            "
          />
        )}
      </div>

      {/* Content Section */}
      <div className="flex-1 flex flex-col justify-between">
        <div className="space-y-1">
          {product.sale_price ? (
            <div>
              <p className="text-xs text-red-600 break-words sm:text-sm">
                Sale: {product.sale_price}
              </p>
              {product.was_price && (
                <p className="text-[10px] text-gray-500 line-through">
                  Was: {product.was_price}
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-gray-800 break-words sm:text-sm">
              {product.price || "N/A"}
            </p>
          )}
          <p className="text-[11px] font-medium text-gray-800 break-words sm:text-xs">
            {product.title || "No Title"}
          </p>
          {product.brand && (
            <p className="text-[10px] text-gray-500 break-words">
              {product.brand}
            </p>
          )}
        </div>
        <div className="flex items-start gap-2 text-[10px] text-gray-500 mt-2 sm:gap-2 sm:text-[10px]">
          <div className="w-10 h-6 bg-gray-300 rounded-sm border border-gray-400 flex-shrink-0" />
          <div className="leading-tight">
            <p className="break-words">{product.store_distance_km || "N/A"}</p>
            <p className="text-[9px] text-gray-500 break-words sm:text-[8px]">
              {product.store_address || "No Address"}
            </p>
          </div>
        </div>
      </div>

      {/* Add/Check Button */}
      <motion.button
        onClick={handleAddToList}
        className="
          absolute
          top-3 right-3
          p-1
          rounded-full
          border-0
          bg-transparent
        "
        whileTap={{ scale: 0.9 }}
      >
        <motion.div
          animate={{ rotate: isInList ? 360 : 0 }}
          transition={{ duration: 0.5 }}
        >
          {isInList ? (
            <FontAwesomeIcon
              icon={faCheckCircle}
              className="text-green-500"
              size="lg"
            />
          ) : (
            <FontAwesomeIcon
              icon={faCirclePlus}
              className="text-black"
              size="lg"
            />
          )}
        </motion.div>
      </motion.button>
    </motion.div>
  );
};

export default Card;
