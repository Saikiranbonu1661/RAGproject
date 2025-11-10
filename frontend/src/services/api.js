import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = {
  // Session management
  createSession: () => {
    return axios.post(`${API_BASE_URL}/api/session/create`);
  },

  getSession: (sessionId) => {
    return axios.get(`${API_BASE_URL}/api/session/${sessionId}`);
  },

  deleteSession: (sessionId) => {
    return axios.delete(`${API_BASE_URL}/api/session/${sessionId}`);
  },

  // Document management
  uploadDocument: (file, sessionId) => {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    return axios.post(`${API_BASE_URL}/api/documents/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params: {
        session_id: sessionId
      }
    });
  },

  deleteDocument: (sessionId, documentId) => {
    return axios.delete(`${API_BASE_URL}/api/session/${sessionId}/documents/${documentId}`);
  },

  // Query
  query: (question, sessionId) => {
    return axios.post(`${API_BASE_URL}/api/query`, {
      question,
      session_id: sessionId
    });
  }
};

export default api;

