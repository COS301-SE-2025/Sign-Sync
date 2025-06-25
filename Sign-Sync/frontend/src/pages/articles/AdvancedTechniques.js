import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiZap, FiCode, FiCpu, FiSettings } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const AdvancedTechniques = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Advanced Techniques
          </h1>

          <div className={`p-6 rounded-xl border mb-8 
            ${isDarkMode ? "bg-blue-950 border-blue-800" : "bg-blue-50 border-blue-100"}`}>
            <h2 className="text-xl font-semibold text-blue-500 mb-2">Power User Guide</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Master SignSync's advanced features to boost your productivity
            </p>
          </div>

          <div className="space-y-6">
            {/* Shortcuts Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiZap className="mr-2 text-yellow-500" />
                Keyboard Shortcuts
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    Translation Mode
                  </h3>
                  <ul className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} space-y-2`}>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Ctrl+Space</span> - Toggle between sign/speech</li>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Alt+C</span> - Camera on/off</li>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Alt+M</span> - Microphone on/off</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    Navigation
                  </h3>
                  <ul className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} space-y-2`}>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Ctrl+1</span> - Go to Translator</li>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Ctrl+2</span> - Go to Dashboard</li>
                    <li><span className={`font-mono px-2 py-1 rounded ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>Ctrl+3</span> - Go to Settings</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* API Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiCode className="mr-2 text-purple-500" />
                API Integration
              </h2>
              <div className="space-y-4">
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  Connect SignSync with other applications using our REST API:
                </p>
                <div className="bg-gray-800 text-green-400 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm">
                    {`// Sample API call\n`}
                    {`fetch('https://api.signsync.com/v1/translate', {\n`}
                    {`  method: 'POST',\n`}
                    {`  headers: { 'Authorization': 'Bearer YOUR_API_KEY' },\n`}
                    {`  body: JSON.stringify({ text: 'Hello world' })\n`}
                    {`})`}
                  </code>
                </div>
                <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                  Get your API key from Account Settings → Developer Options
                </p>
              </div>
            </div>

            {/* Performance Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiCpu className="mr-2 text-blue-500" />
                Performance Optimization
              </h2>
              <div className="space-y-3">
                <h3 className={`${isDarkMode ? "text-white" : "text-gray-800"} font-medium`}>Hardware Acceleration</h3>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                  Enable GPU processing in Settings → Performance to reduce CPU usage by up to 40%.
                </p>

                <h3 className={`${isDarkMode ? "text-white" : "text-gray-800"} font-medium`}>Quality Presets</h3>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                  Choose between "Speed", "Balanced", or "Accuracy" modes based on your needs.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AdvancedTechniques;