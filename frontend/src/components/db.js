import { openDB } from "idb";

export const initDB = async () => {
  return openDB("supermarketLocator", 1, {
    upgrade(db) {
      if (!db.objectStoreNames.contains("searches")) {
        db.createObjectStore("searches", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("scrapedStores")) {
        db.createObjectStore("scrapedStores", { keyPath: "address" });
      }
    },
  });
};
