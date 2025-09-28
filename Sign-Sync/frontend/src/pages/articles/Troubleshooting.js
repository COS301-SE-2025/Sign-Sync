import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import PreferenceManager from "../../components/PreferenceManager";
import { FiCamera, FiMic, FiWifi, FiAlertTriangle, FiMessageSquare } from 'react-icons/fi';

const Troubleshooting = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    // <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
    <section
      className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
      style={{
        background: isDarkMode
          ? "linear-gradient(135deg, #080C1A, #172034)"
          : "#f5f5f5",
      }}
    >
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Troubleshooting Guide</h1>
          
          <div className="space-y-6">
            {/* Connection Issues */}
            <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiWifi className="mr-2 text-blue-600" />
                Connection Problems
              </h2>
              <div className="space-y-4">
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Symptoms</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• "No internet connection" error</li>
                    <li>• Frequent disconnections</li>
                    <li>• Slow translation speeds</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Solutions</h3>
                  <ol className={`text-sm space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>1. Check your network connection</li>
                    <li>2. Restart your router</li>
                    <li>3. Try switching between WiFi/mobile data</li>
                    <li>4. Contact your ISP if problems persist</li>
                  </ol>
                </div>
              </div>
            </div>

            {/* Device Compatibility */}
            <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiAlertTriangle className="mr-2 text-yellow-600" />
                Device Issues
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className={`font-medium mb-2 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    <FiCamera className="mr-2" />
                    Camera Problems
                  </h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Ensure camera permissions are granted</li>
                    <li>• Clean camera lens</li>
                    <li>• Test with another app</li>
                    <li>• Update device drivers</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    <FiMic className="mr-2" />
                    Microphone Issues
                  </h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Check microphone permissions</li>
                    <li>• Ensure correct input device is selected</li>
                    <li>• Reduce background noise</li>
                    <li>• Test microphone volume</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Support Contact */}
            <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiMessageSquare className="mr-2 text-green-600" />
                Need More Help?
              </h2>
              <div className={`space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <p><span className="font-medium">Email:</span> support@signsync.com (24hr response)</p>
                <p><span className="font-medium">Live Chat:</span> Available Mon-Fri, 9AM-5PM EST</p>
                <p><span className="font-medium">Emergency:</span> +1 (555) 123-4567 (Critical outages only)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Troubleshooting;