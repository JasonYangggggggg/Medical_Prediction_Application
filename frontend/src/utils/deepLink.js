// Deep-link utility for launching the desktop scraper app
import { scraperApi } from './api.js';

export const deepLinkUtils = {
  // Enhanced location detection with caching
  async getUserLocation() {
    try {
      // Try to get from backend first
      const location = await scraperApi.getLocation();
      if (location && location.lat && location.lon) {
        // Cache location in localStorage
        localStorage.setItem('userLocation', JSON.stringify(location));
        return location;
      }
    } catch (error) {
      console.log('Backend location detection failed:', error);
    }

    // Fallback to cached location
    try {
      const cached = localStorage.getItem('userLocation');
      if (cached) {
        const location = JSON.parse(cached);
        if (location && location.lat && location.lon) {
          return location;
        }
      }
    } catch (error) {
      console.log('Cached location invalid:', error);
    }

    return null;
  },

  // Check if the desktop app is installed by trying to launch it
  async isAppInstalled() {
    try {
      // Try to open a test deep-link
      const testLink = 'scraper://test';
      window.location.href = testLink;
      return true;
    } catch (error) {
      return false;
    }
  },

  // Launch scraper with deep-link
  async launchScraper(query) {
    console.log('🚀 Launching scraper for query:', query);
    
    // Try REST API first (more reliable)
    try {
      const result = await scraperApi.scrapeDeals(query);
      console.log('✅ Scraper started via REST API');
      return this.pollForResults(query);
    } catch (error) {
      console.log('REST API failed, trying deep-link:', error.message);
    }
    
    // Fallback to deep-link
    const encodedQuery = encodeURIComponent(query);
    const deepLink = `scraper://search?query=${encodedQuery}`;
    
    console.log(`🔗 Launching scraper with deep-link: ${deepLink}`);
    
    // Try to open the deep-link
    window.location.href = deepLink;
    
    // Return a promise that resolves when we expect the scraping to be done
    return new Promise((resolve) => {
      // Start polling for results after a short delay
      setTimeout(() => {
        console.log('📊 Starting to poll for results...');
        this.pollForResults(query, resolve);
      }, 2000);
    });
  },

  // Launch map lookup with deep-link
  launchMapLookup(query) {
    const encodedQuery = encodeURIComponent(query);
    const deepLink = `scraper://map?query=${encodedQuery}`;
    
    console.log(`🗺️ Launching map lookup with deep-link: ${deepLink}`);
    window.location.href = deepLink;
  },

  // Poll for scraping results (wrapper function)
  async pollForResults(query) {
    return new Promise((resolve) => {
      this.pollForResultsInternal(query, resolve);
    });
  },

  // Poll for scraping results (internal implementation)
  async pollForResultsInternal(query, resolve, attempt = 0, maxAttempts = 30) {
    const pollInterval = 1000; // 1 second
    const maxWaitTime = 30000; // 30 seconds max
    
    if (attempt >= maxAttempts) {
      console.error('Polling timed out after', maxWaitTime / 1000, 'seconds');
      resolve({ error: 'Timeout waiting for results' });
      return;
    }

    try {
      console.log(`📡 Poll attempt ${attempt + 1}: fetching results`);
      
      // Try to fetch results from the local server (if running)
      const data = await scraperApi.getResults();
      console.log('📊 Poll result:', data ? 'Data received' : 'No data');
      
      if (data && data.query?.toLowerCase() === query.toLowerCase()) {
        console.log('✅ Deep-link scraping completed:', data);
        resolve(data);
      } else {
        // Keep polling
        setTimeout(() => {
          this.pollForResultsInternal(query, resolve, attempt + 1, maxAttempts);
        }, pollInterval);
      }
    } catch (error) {
      console.warn(`Poll attempt ${attempt + 1} failed:`, error.message);
      
      // Keep polling on error (app might still be starting)
      setTimeout(() => {
        this.pollForResultsInternal(query, resolve, attempt + 1, maxAttempts);
      }, pollInterval);
    }
  },

  // Fallback: show instructions for manual installation
  showInstallInstructions() {
    const instructions = `
To use the deep-link functionality:

1. Make sure the BargainBee desktop app is installed
2. Run the app once to register the 'scraper://' protocol
3. The app will handle searches automatically when you click search

If the app isn't installed, download it from the Downloads page.
    `;
    
    alert(instructions);
  }
};

// Make it globally available
window.deepLink = deepLinkUtils;
