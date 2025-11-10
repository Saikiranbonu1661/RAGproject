import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import api from './services/api';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await api.createSession();
        setSessionId(response.data.session_id);
        
        // Add welcome message
        setMessages([{
          role: 'assistant',
          content: 'Hello! Upload documents to get started. I\'ll answer questions based on your uploaded documents.',
          timestamp: new Date().toISOString()
        }]);
      } catch (error) {
        console.error('Error creating session:', error);
      }
    };
    
    initSession();
  }, []);

  // Handle document upload
  const handleUpload = async (files) => {
    setLoading(true);
    
    try {
      const uploadedDocs = [];
      
      for (const file of files) {
        const response = await api.uploadDocument(file, sessionId);
        uploadedDocs.push(response.data.document);
        
        // Update total chunks
        setTotalChunks(response.data.total_chunks);
      }
      
      setDocuments(prev => [...prev, ...uploadedDocs]);
      
      // Add system message
      setMessages(prev => [...prev, {
        role: 'system',
        content: `✅ Successfully uploaded ${uploadedDocs.length} document(s)`,
        timestamp: new Date().toISOString()
      }]);
      
    } catch (error) {
      console.error('Error uploading document:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `❌ Error uploading document: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Handle sending query
  const handleSendMessage = async (question) => {
    if (!question.trim() || documents.length === 0) return;
    
    // Add user message
    const userMessage = {
      role: 'user',
      content: question,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    
    setLoading(true);
    
    try {
      const response = await api.query(question, sessionId);
      
      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        token_usage: response.data.token_usage,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error('Error querying:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `❌ Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Handle document deletion
  const handleDeleteDocument = async (documentId) => {
    try {
      await api.deleteDocument(sessionId, documentId);
      setDocuments(prev => prev.filter(doc => doc.id !== documentId));
      
      setMessages(prev => [...prev, {
        role: 'system',
        content: '🗑️ Document deleted',
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  // Handle clear all
  const handleClearAll = async () => {
    if (window.confirm('Clear all documents and chat history?')) {
      try {
        await api.deleteSession(sessionId);
        
        // Create new session
        const response = await api.createSession();
        setSessionId(response.data.session_id);
        setDocuments([]);
        setTotalChunks(0);
        setMessages([{
          role: 'assistant',
          content: 'Session cleared. Upload documents to start fresh!',
          timestamp: new Date().toISOString()
        }]);
      } catch (error) {
        console.error('Error clearing session:', error);
      }
    }
  };

  return (
    <div className="App">
      <Sidebar
        documents={documents}
        totalChunks={totalChunks}
        onUpload={handleUpload}
        onDeleteDocument={handleDeleteDocument}
        onClearAll={handleClearAll}
        loading={loading}
      />
      <ChatInterface
        messages={messages}
        onSendMessage={handleSendMessage}
        loading={loading}
        hasDocuments={documents.length > 0}
      />
    </div>
  );
}

export default App;

