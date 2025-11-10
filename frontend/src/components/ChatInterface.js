import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatInterface.css';

function ChatInterface({ messages, onSendMessage, loading, hasDocuments }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !loading && hasDocuments) {
      onSendMessage(input);
      setInput('');
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit'
    });
  };

  const formatTokenUsage = (usage) => {
    if (!usage) return null;
    
    let text = `${usage.prompt_tokens} prompt + ${usage.completion_tokens} completion`;
    
    if (usage.reasoning_tokens) {
      text += ` (output: ${usage.output_tokens}, reasoning: ${usage.reasoning_tokens})`;
    }
    
    text += ` = ${usage.total_tokens} total`;
    
    return text;
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>💬 Chat with Your Documents</h1>
        <p>Ask questions about your uploaded documents</p>
      </div>

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.role}`}
          >
            <div className="message-content">
              {message.role === 'user' && (
                <div className="message-avatar user-avatar">👤</div>
              )}
              {message.role === 'assistant' && (
                <div className="message-avatar assistant-avatar">🤖</div>
              )}
              {message.role === 'system' && (
                <div className="message-avatar system-avatar">ℹ️</div>
              )}
              
              <div className="message-body">
                <div className="message-text">
                  {message.role === 'system' ? (
                    message.content
                  ) : (
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  )}
                </div>
                
                {message.token_usage && (
                  <div className="token-usage">
                    🔢 Tokens: {formatTokenUsage(message.token_usage)}
                  </div>
                )}
                
                {message.sources && message.sources.length > 0 && (
                  <details className="sources">
                    <summary>📚 Sources ({message.sources.length})</summary>
                    <div className="sources-list">
                      {message.sources.map((source, idx) => (
                        <div key={idx} className="source-item">
                          <div className="source-title">
                            Source {idx + 1}: {source.source}
                          </div>
                          <div className="source-content">
                            {source.content}
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                
                <div className="message-timestamp">
                  {formatTimestamp(message.timestamp)}
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="message assistant">
            <div className="message-content">
              <div className="message-avatar assistant-avatar">🤖</div>
              <div className="message-body">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              hasDocuments 
                ? "Ask a question about your documents..." 
                : "Upload documents to start chatting..."
            }
            className="chat-input"
            disabled={loading || !hasDocuments}
          />
          <button
            type="submit"
            className="send-button"
            disabled={loading || !hasDocuments || !input.trim()}
          >
            <span className="send-icon">➤</span>
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatInterface;

