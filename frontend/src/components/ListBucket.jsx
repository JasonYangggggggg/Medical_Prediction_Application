import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTrash } from "@fortawesome/free-solid-svg-icons";
import { motion, AnimatePresence } from "framer-motion";

const ListBucket = ({
  bucket,
  highlightedBucket,
  deleteBucket,
  expandedItems,
  toggleExpand,
  deleteItem,
  handleDragEnter,
  handleDragLeave,
  onDragEnd,
  handleQuantityChange,
}) => {
  return (
    <motion.div
      key={bucket.id}
      data-bucket-id={bucket.id}
      className="bucket border-2 border-gray-300 p-4 w-[400px] bg-gray-50 rounded-lg relative"
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={() => handleDragEnter(bucket.id)}
      onDragLeave={() => handleDragLeave(bucket.id)}
      animate={{
        scale: highlightedBucket === bucket.id ? 1.05 : 1,
        borderColor: highlightedBucket === bucket.id ? "gold" : "gray",
      }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 20,
      }}
    >
      <h3 className="font-medium text-md">{bucket.name}</h3>
      {bucket.id !== "default" && (
        <button
          onClick={() => deleteBucket(bucket.id)}
          className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
        >
          <FontAwesomeIcon icon={faTrash} />
        </button>
      )}
      <motion.div className="space-y-4 mt-4">
        <AnimatePresence>
          {bucket.items.map((item) => (
            <ListItem
              key={item.id}
              item={item}
              bucketId={bucket.id}
              expandedItems={expandedItems}
              toggleExpand={toggleExpand}
              deleteItem={deleteItem}
              onDragEnd={onDragEnd}
              handleQuantityChange={handleQuantityChange}
            />
          ))}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};

export default ListBucket;
