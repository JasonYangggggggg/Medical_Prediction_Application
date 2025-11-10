import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTrash, faExpandAlt, faCompressAlt } from "@fortawesome/free-solid-svg-icons";
import { motion } from "framer-motion";

const ListItem = ({
  item,
  bucketId,
  expandedItems,
  toggleExpand,
  deleteItem,
  onDragEnd,
  handleQuantityChange,
}) => {
  const details = item.details?.details || {};
  const isOnSale = details.sale_price;
  const isExpanded = expandedItems.includes(item.id);

  return (
    <motion.div
      key={item.id}
      drag
      dragConstraints={{ top: 0, bottom: 0, left: 0, right: 0 }}
      dragSnapToOrigin={false}
      onDragEnd={(event) => onDragEnd(event, item, bucketId)}
      whileDrag={{
        scale: 1.05,
        boxShadow: "0px 5px 10px rgba(0,0,0,0.1)",
        zIndex: 1000,
      }}
      className="bg-white border border-gray-300 rounded-lg shadow-md p-4 relative"
    >
      <motion.div
        animate={{
          height: isExpanded ? "auto" : "60px",
        }}
        transition={{
          duration: 0.3,
        }}
        className="overflow-hidden"
      >
        <div className="flex items-center space-x-3">
          <img
            src={details.image}
            alt={details.title || "Product"}
            className="w-12 h-12 object-cover rounded-lg"
          />
          <div>
            <p className="font-bold text-gray-800">
              {details.title || "Unnamed Item"}
            </p>
            {isOnSale ? (
              <p className="font-bold text-red-600">
                {details.sale_price}{" "}
                <span className="line-through text-gray-500">
                  {details.was_price}
                </span>
              </p>
            ) : (
              <p className="text-gray-600">{details.price || "N/A"}</p>
            )}
          </div>
        </div>
        {isExpanded && (
          <div className="mt-3 max-h-[200px] overflow-y-auto p-2">
            {Object.entries(details).map(([key, value]) => (
              <p key={key} className="text-sm text-gray-500 capitalize">
                {key}: {value || "N/A"}
              </p>
            ))}
          </div>
        )}
      </motion.div>

      <div className="mt-2">
        <label className="text-sm text-gray-600 mr-2">Quantity:</label>
        <input
          type="number"
          value={item.quantity || 1}
          onChange={(e) =>
            handleQuantityChange(item.id, bucketId, parseInt(e.target.value))
          }
          className="w-16 p-1 border border-gray-300 rounded"
          min="1"
        />
      </div>

      <div className="absolute top-2 right-2 flex space-x-2">
        <button
          onClick={() => toggleExpand(item.id)}
          className="text-gray-500 hover:text-gray-700"
          title={isExpanded ? "Collapse" : "Expand"}
        >
          <FontAwesomeIcon
            icon={isExpanded ? faCompressAlt : faExpandAlt}
          />
        </button>
        <button
          onClick={() => deleteItem(item.id, bucketId)}
          className="text-gray-500 hover:text-gray-700"
          title="Delete item"
        >
          <FontAwesomeIcon icon={faTrash} />
        </button>
      </div>
    </motion.div>
  );
};

export default ListItem;
