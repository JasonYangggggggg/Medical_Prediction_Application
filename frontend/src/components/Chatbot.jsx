import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";

// 1) Import Font Awesome Icon components
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
// 2) Import the specific icons you want
import { faRotateLeft, faTimes } from "@fortawesome/free-solid-svg-icons";

import defaultIcon from "../assets/default.png";
import onGrabIcon from "../assets/on-grab.png";
import onCollideIcon from "../assets/on-collide.png";
import onOpenIcon from "../assets/on-open.png";
import onTalkIcon from "../assets/on-talk.gif"; // "talking" GIF icon
import { chatWithLlama } from "./llamaApi";

const IDLE_TIMEOUT = 20000; // 20 seconds

const Chatbot = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [currentIcon, setCurrentIcon] = useState(defaultIcon);
  const [isBouncing, setIsBouncing] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  // Whether we're waiting for the server to respond (spinner)
  const [isLoading, setIsLoading] = useState(false);

  // Whether the bot is actively streaming/typing its response
  const [isStreaming, setIsStreaming] = useState(false);

  const constraintsRef = useRef(null);
  const dragStartTime = useRef(null);

  const DRAG_THRESHOLD = 500;
  const TYPING_SPEED = 30; // ms between each character
  const chatContentRef = useRef(null);

  // We'll store an idle timer ref to clear/reset as user interacts
  const idleTimerRef = useRef(null);

  // ------------------------------------------------------------
  // IDLE TIMER LOGIC
  // ------------------------------------------------------------
  function stopIdleTimer() {
    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  }

  function startIdleTimer() {
    stopIdleTimer();
    idleTimerRef.current = setTimeout(() => {
      // Only revert if not streaming
      if (!isStreaming) {
        setCurrentIcon(defaultIcon);
      }
    }, IDLE_TIMEOUT);
  }

  function resetIdleTimer() {
    startIdleTimer();
  }

  // Start the idle timer on mount
  useEffect(() => {
    startIdleTimer();
    return () => stopIdleTimer();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Whenever messages change, scroll to the bottom
  useEffect(() => {
    if (chatContentRef.current) {
      chatContentRef.current.scrollTop = chatContentRef.current.scrollHeight;
    }
  }, [messages]);

  // Dynamically set the bounding box for dragging
  useEffect(() => {
    if (constraintsRef.current) {
      constraintsRef.current.style.width = `${window.innerWidth}px`;
      constraintsRef.current.style.height = `${window.innerHeight}px`;
    }
  }, []);

  // ------------------------------------------------------------
  // DRAG & CLICK EVENTS (all call resetIdleTimer)
  // ------------------------------------------------------------
  const handleDragStart = () => {
    resetIdleTimer();
    dragStartTime.current = Date.now();
    setCurrentIcon(onGrabIcon);
  };

  const handleDragEnd = () => {
    resetIdleTimer();
    dragStartTime.current = null;
    // Only revert if chat is not open & not streaming
    if (!isChatOpen && !isStreaming) {
      setCurrentIcon(defaultIcon);
    }
    setIsBouncing(false);
  };

  const handleChatboxClick = (e) => {
    resetIdleTimer();
    e.stopPropagation();
  };

  const handleClick = (e) => {
    resetIdleTimer();
    e.stopPropagation();

    const dragDuration = dragStartTime.current
      ? Date.now() - dragStartTime.current
      : 0;

    if (dragDuration < DRAG_THRESHOLD) {
      setIsChatOpen((prev) => !prev);
      if (!isChatOpen) {
        setCurrentIcon(onOpenIcon);
      } else {
        // Only revert if we’re not streaming
        if (!isStreaming) {
          setCurrentIcon(defaultIcon);
        }
      }
    }
  };

  const handleMotionStart = () => {
    resetIdleTimer();
    setIsBouncing(true);
    if (!isStreaming) {
      setCurrentIcon(onCollideIcon);
    }
  };

  const handleMotionEnd = () => {
    resetIdleTimer();
    setIsBouncing(false);
    if (!isStreaming) {
      setCurrentIcon(onGrabIcon);
    }
  };

  // ------------------------------------------------------------
  // CHAT LOGIC
  // ------------------------------------------------------------
  const handleUserInputChange = (e) => {
    resetIdleTimer();
    setUserInput(e.target.value);
  };

  const handleSendMessage = async () => {
    resetIdleTimer();
    if (!userInput.trim()) return;

    // 1) Add user's message
    const newMessages = [...messages, { role: "user", content: userInput }];
    setMessages(newMessages);
    setUserInput("");

    // 2) Show spinner while waiting for server
    setIsLoading(true);

    try {
      // 3) Get the full response from Llama
      // const fullBotMessage = await chatWithLlama(userInput);
      const fullBotMessage = "Hang tight! Im coming to assist you soon (in v2)!";
      // 4) Remove spinner
      setIsLoading(false);

      // 5) Add empty assistant message to fill via streaming
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      // 6) Start streaming
      setIsStreaming(true);
      setCurrentIcon(onTalkIcon); // Switch to "talking" GIF icon

      simulateStream(fullBotMessage);
    } catch (error) {
      console.error("Error:", error);
      setIsLoading(false);
    }
  };

  // "Simulated" streaming, typed character by character
  const simulateStream = (botMessage) => {
    let index = 0;
    const intervalId = setInterval(() => {
      setMessages((prevMessages) => {
        if (!prevMessages.length) {
          clearInterval(intervalId);
          return prevMessages;
        }
        const lastIndex = prevMessages.length - 1;
        const lastMessage = prevMessages[lastIndex];
        if (lastMessage.role !== "assistant") {
          clearInterval(intervalId);
          return prevMessages;
        }
        // Append the next character
        const updatedContent = lastMessage.content + botMessage.charAt(index);
        const updatedMessage = { ...lastMessage, content: updatedContent };

        const newMessages = [...prevMessages];
        newMessages[lastIndex] = updatedMessage;

        // If we've reached the end of the message, stop streaming
        if (index >= botMessage.length - 1) {
          clearInterval(intervalId);
          setIsStreaming(false); // Done typing
          setCurrentIcon(defaultIcon);
        } else {
          index++;
        }
        return newMessages;
      });
    }, TYPING_SPEED);
  };

  // ------------------------------------------------------------
  // RESET CONVERSATION BUTTON
  // ------------------------------------------------------------
  const handleResetConversation = (e) => {
    e.stopPropagation(); // don't toggle chat
    resetIdleTimer();
    setMessages([]);
    // If not streaming, revert to default icon
    if (!isStreaming) {
      setCurrentIcon(defaultIcon);
    }
  };

  // ------------------------------------------------------------
  // RENDER
  // ------------------------------------------------------------
  return (
    <div
      ref={constraintsRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 999999 }}
    >
      <motion.div
        drag
        dragConstraints={constraintsRef}
        dragElastic={0.6}
        dragTransition={{
          bounceStiffness: 300,
          bounceDamping: 15,
        }}
        animate={{
          y: [0, -10, 0],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          repeatType: "reverse",
          ease: "easeInOut",
        }}
        whileTap={{ scale: 1.1 }}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onDrag={handleMotionStart}
        onDragTransitionEnd={handleMotionEnd}
        className="absolute"
        style={{
          pointerEvents: "auto",
          cursor: "grab",
          bottom: "90px",
          right: "140px",
          position: "fixed",
          zIndex: 999999, // Ensure on top
        }}
        onClick={handleClick}
      >
        {/* Draggable transparent circle */}
        <div
          className="w-16 h-16 rounded-full absolute bg-transparent"
          style={{ pointerEvents: "auto" }}
        ></div>

        {/* Icon */}
        <img
          src={currentIcon}
          alt="Chatbot Icon"
          className="w-16 h-16 object-contain pointer-events-none"
        />

        {/* Chatbox */}
        {isChatOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="absolute top-0 left-[calc(100%+10px)] w-[300px] h-[400px]
                       bg-white rounded-lg shadow-lg border-2 border-gray-200
                       rounded-tr-[30px] rounded-bl-[30px] border-t-0
                       flex flex-col"
            style={{ zIndex: 999999 }}
            onClick={handleChatboxClick}
          >
            {/* Chat Header */}
            <div className="flex-shrink-0 flex justify-between items-center bg-gray-100 p-2 rounded-t-lg">
              <div className="flex items-center">
                <div className="w-4 h-4 bg-gray-400 rounded-full mr-2"></div>
                <h2 className="text-sm font-semibold text-gray-700">
                  Chatbot Name
                </h2>
              </div>

              {/* Top-right Buttons (Reset & Close) */}
              <div className="flex items-center space-x-3">
                {/* Reset button with Font Awesome icon */}
                <button
                  className="text-gray-500 hover:text-green-600"
                  onClick={handleResetConversation}
                >
                  <FontAwesomeIcon icon={faRotateLeft} />
                </button>

                {/* Close button with Font Awesome icon */}
                <button
                  className="text-gray-500 hover:text-red-600"
                  onClick={(e) => {
                    e.stopPropagation();
                    resetIdleTimer();
                    setIsChatOpen(false);
                    if (!isStreaming) {
                      setCurrentIcon(defaultIcon);
                    }
                  }}
                >
                  <FontAwesomeIcon icon={faTimes} />
                </button>
              </div>
            </div>

            {/* Chat Content (flex-1, scrollable) */}
            <div
              ref={chatContentRef}
              className="flex-1 overflow-y-auto mt-4 px-1"
            >
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`mb-2 ${msg.role === "user" ? "ml-auto" : ""}`}
                >
                  <div
                    className={`p-2 rounded-lg ${
                      msg.role === "user" ? "bg-blue-100" : "bg-gray-100"
                    }`}
                  >
                    {msg.role === "assistant" ? (
                      <ReactMarkdown className="text-gray-700 text-sm whitespace-pre-wrap">
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      <p className="text-gray-700 text-sm whitespace-pre-wrap">
                        {msg.content}
                      </p>
                    )}
                  </div>
                </div>
              ))}

              {/* Show spinner if isLoading is true */}
              {isLoading && (
                <div className="mb-2">
                  <div className="flex items-center p-2 rounded-lg bg-gray-100">
                    <p className="text-gray-700 text-sm mr-2">Thinking...</p>
                    <div className="loader border-4 border-gray-200 rounded-full h-5 w-5 border-t-blue-500 animate-spin"></div>
                  </div>
                </div>
              )}
            </div>

            {/* Chat Input */}
            <div className="flex-shrink-0 mt-4">
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg
                           focus:outline-none focus:ring focus:ring-blue-500"
                placeholder="Type a message..."
                value={userInput}
                onChange={handleUserInputChange}
                onKeyPress={(e) => {
                  if (e.key === "Enter") {
                    handleSendMessage();
                  }
                }}
              />
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default Chatbot;
