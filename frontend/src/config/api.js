// API configuration for connecting to backend
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const getApiUrl = (path) => {
  return `${API_BASE_URL}${path}`;
};

console.log('API Base URL:', API_BASE_URL);
