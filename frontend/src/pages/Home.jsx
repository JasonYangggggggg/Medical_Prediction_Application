import React, { useState, useEffect, useRef } from 'react';
import { FaUserCircle, FaRobot } from 'react-icons/fa';

const Home = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageListRef = useRef(null);

  useEffect(() => {
    const storedChat = sessionStorage.getItem('chatHistory');
    if (storedChat) {
      setChatHistory(JSON.parse(storedChat));
    }
  }, []);

  
  useEffect(() => {
    sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
  }, [chatHistory]);

  
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: message };
    setChatHistory((prevHistory) => [...prevHistory, userMessage]);
    setMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(
        `http://localhost:8000/getAnswer?message=${encodeURIComponent(message)}`
      );

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const botResponseText = await response.text();
      const botMessage = { sender: 'bot', text: botResponseText };
      
      setChatHistory((prevHistory) => [...prevHistory, botMessage]);

    } catch (error) {
      console.error('Error fetching answer:', error);
      const errorMessage = { 
        sender: 'bot', 
        text: 'Sorry, something went wrong. Please try again.' 
      };
      setChatHistory((prevHistory) => [...prevHistory, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 p-4">
      <div className="flex flex-col w-full max-w-xl mx-auto bg-white rounded-lg shadow-xl h-full">
        <div className="p-4 bg-gray-50 border-b border-gray-200 text-center font-bold text-lg text-gray-800">
          Covid-19 Medical Assistant
        </div>

        <div ref={messageListRef} className="flex-1 p-6 overflow-y-auto space-y-4">
          {chatHistory.map((msg, index) => (
            <div
              key={index}
              className={`flex items-start ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
             
              {msg.sender === 'bot' && (
                <FaRobot className="w-8 h-8 text-blue-600 mr-3 flex-shrink-0" />
              )}
              
             
              <div
                className={`max-w-[75%] p-3 rounded-xl text-sm leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-gray-200 text-gray-800 rounded-bl-none'
                }`}
              >
                {msg.text}
              </div>

             
              {msg.sender === 'user' && (
                <FaUserCircle className="w-8 h-8 text-gray-400 ml-3 flex-shrink-0" />
              )}
            </div>
          ))}

        
          {isLoading && (
            <div className="flex items-start justify-start">
              <FaRobot className="w-8 h-8 text-blue-600 mr-3 flex-shrink-0" />
              <div className="max-w-[75%] p-3 rounded-xl bg-gray-200 text-gray-500 italic rounded-bl-none">
                Typing...
              </div>
            </div>
          )}
        </div>

        
        <form onSubmit={handleSubmit} className="flex p-4 border-t border-gray-200 bg-gray-50">
          <input
            type="text"
            className="flex-1 p-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="ml-3 px-6 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold"
            disabled={isLoading}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default Home;