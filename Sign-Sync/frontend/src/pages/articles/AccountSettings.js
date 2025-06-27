import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiUser, FiEye, FiMoon, FiAward, FiGlobe } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const AccountSettings = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Account Settings Guide
          </h1>

          <div className="space-y-6">
            {/* Accessibility */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiEye className="mr-2 text-purple-600" />
                Accessibility Settings
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Display Options</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Light/Dark mode toggle</li>
                    <li>• Font size adjustment</li>
                    <li>• High contrast mode</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Avatar Customization</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Change avatar style</li>
                    <li>• Adjust animation speed</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Advanced Settings */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiAward className="mr-2 text-green-600" />
                Advanced Options
              </h2>
              <div className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} space-y-3`}>
                <p><span className="font-medium">Account Deletion:</span> Permanently remove your account</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AccountSettings;