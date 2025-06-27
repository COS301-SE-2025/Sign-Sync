import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiShield, FiEyeOff, FiDatabase, FiUser } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const PrivacySettings = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Privacy Settings</h1>

          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiShield className="mr-2 text-blue-600" />
              Privacy Controls
            </h2>

            <div className="space-y-6">
              <div className="flex items-start">
                <div className="bg-blue-100 p-2 rounded-lg mr-4">
                  <FiEyeOff className="text-blue-600" />
                </div>
                <div>
                  <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Data Collection</h3>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                    Choose what usage data we collect to improve our services
                  </p>
                </div>
              </div>

              <div className="flex items-start">
                <div className="bg-purple-100 p-2 rounded-lg mr-4">
                  <FiDatabase className="text-purple-600" />
                </div>
                <div>
                  <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Storage Duration</h3>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                    Set how long we keep your translation history
                  </p>
                </div>
              </div>

              <div className="flex items-start">
                <div className="bg-green-100 p-2 rounded-lg mr-4">
                  <FiUser className="text-green-600" />
                </div>
                <div>
                  <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Third-Party Sharing</h3>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                    Control whether we share anonymized data with partners
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className={`p-6 rounded-xl border ${isDarkMode ? "bg-blue-900 border-blue-700" : "bg-blue-50 border-blue-100"}`}>
            <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Our Privacy Commitment</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>
              We never sell your personal data. All information is encrypted and protected.
            </p>
            <a
              href="/privacy-policy"
              className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
            >
              View Full Policy
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PrivacySettings;