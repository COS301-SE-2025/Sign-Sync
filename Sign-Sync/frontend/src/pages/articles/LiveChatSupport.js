import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiMessageSquare, FiUser, FiZap } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const LiveChatSupport = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    // <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
    <section
      className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
      style={{
        background: isDarkMode
          ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
          : "linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)",
      }}
    >
      <SideNavbar />

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Live Chat Support
          </h1>

          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-start mb-6">
              <div className={`p-3 rounded-full mr-4 ${isDarkMode ? "bg-green-800" : "bg-green-100"}`}>
                <FiMessageSquare className="text-green-600 text-xl" />
              </div>
              <div>
                <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>Real-Time Assistance</h2>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>Get immediate help from our support team</p>
              </div>
            </div>

            <div className="space-y-6">
              {/* Availability */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Availability</h3>
                <div className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-2`}>
                    <span className="font-medium">Monday-Friday:</span> 9:00 AM - 5:00 PM EST
                  </p>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <span className="font-medium">Saturday-Sunday:</span> Closed
                  </p>
                </div>
              </div>

              {/* How to Access */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>How to Access</h3>
                <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  <li>Click the chat icon in the bottom-right corner</li>
                  <li>Sign in with your account</li>
                  <li>Describe your issue to the bot</li>
                  <li>Wait for a human agent to join</li>
                </ol>
              </div>

              {/* Pro Tips */}
              <div className={`p-4 rounded-lg border ${isDarkMode ? "bg-blue-900 border-blue-800" : "bg-blue-50 border-blue-100"}`}>
                <h3 className={`text-lg font-medium mb-2 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                  <FiZap className="mr-2 text-blue-600" />
                  Pro Tips
                </h3>
                <ul className={`list-disc pl-5 space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  <li>Have your account email ready</li>
                  <li>Chat sessions timeout after 15 minutes of inactivity</li>
                  <li>You'll receive a transcript via email</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LiveChatSupport;