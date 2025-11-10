import React, { useState } from "react";
import LoginPage from "../pages/LoginPage";
import RegisterPage from "../pages/RegisterPage";
import TokenVerificationPage from "../pages/TokenVerificationPage"; // Import the verification page

const AuthFlow = ({ onAuthenticated }) => {
  const [currentPage, setCurrentPage] = useState("login"); // Tracks the current page

  const handleContinueAsGuest = () => {
    // Mark the user as a guest
    localStorage.setItem("isGuest", "true"); // Save guest status in localStorage
    onAuthenticated(); // Redirect the user to the app
  };

  const renderPage = () => {
    switch (currentPage) {
      case "login":
        return (
          <LoginPage
            onAuthenticated={onAuthenticated}
            onSwitchToRegister={() => setCurrentPage("register")} // Navigate to register
            onContinueAsGuest={handleContinueAsGuest} // Handle "Continue as Guest"
          />
        );
      case "register":
        return (
          <RegisterPage
            onSwitchToLogin={() => setCurrentPage("login")} // Navigate to login
            onRegisterSuccess={() => setCurrentPage("verify")} // Navigate to token verification
          />
        );
      case "verify":
        return (
          <TokenVerificationPage
            onVerified={() => setCurrentPage("login")} // Navigate back to login after verification
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 space-y-4">
      {renderPage()}
      {/* Continue as Guest Button (Only shown on the login page) */}
      {currentPage === "login" && (
        <button
          onClick={handleContinueAsGuest}
          className="mt-6 bg-yellow-500 hover:bg-yellow-600 text-white font-bold px-6 py-3 rounded-lg shadow-md transition-all"
        >
          Continue as Guest
        </button>
      )}
    </div>
  );
};

export default AuthFlow;
