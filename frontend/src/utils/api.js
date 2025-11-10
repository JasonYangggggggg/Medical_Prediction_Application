const API_KEY = "TuPxJK21034";
const BASE_URL = "http://localhost:3334";

const apiHeaders = {
  'Authorization': `Bearer ${API_KEY}`,
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json'
};

// Helper function to handle API responses
const handleResponse = async (response, endpoint) => {
  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`${endpoint} failed (${response.status}): ${errorText}`);
  }
  return response.json();
};

export const scraperApi = {
  async getLocation() {
    try {
      const response = await fetch(`${BASE_URL}/location`, {
        headers: apiHeaders
      });
      return handleResponse(response, 'Location API');
    } catch (error) {
      console.error('Failed to get location:', error.message);
      throw error;
    }
  },

  async scrapeDeals(query) {
    try {
      const response = await fetch(`${BASE_URL}/scrape?query=${encodeURIComponent(query)}`, {
        headers: apiHeaders
      });
      const result = await handleResponse(response, 'Scrape API');
      
      // Just return the confirmation, actual results will be in results.json
      return result;
    } catch (error) {
      console.error('Failed to scrape deals:', error.message);
      throw error;
    }
  },

  async getMap(query) {
    try {
      const response = await fetch(`${BASE_URL}/map?query=${encodeURIComponent(query)}`, {
        headers: apiHeaders
      });
      return handleResponse(response, 'Map API');
    } catch (error) {
      console.error('Failed to get map data:', error.message);
      throw error;
    }
  },

  async getResults() {
    try {
      const response = await fetch(`${BASE_URL}/results.json`, {
        headers: apiHeaders
      });
      return handleResponse(response, 'Results API');
    } catch (error) {
      console.error('Failed to get results:', error.message);
      throw error;
    }
  },

  async healthCheck() {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error.message);
      return false;
    }
  }
};