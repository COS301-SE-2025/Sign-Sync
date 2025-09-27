import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiPhone, FiClock, FiGlobe } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const PhoneSupport = () => {
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
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Phone Support</h1>

          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-start mb-6">
              <div className="bg-red-100 p-3 rounded-full mr-4">
                <FiPhone className="text-red-600 text-xl" />
              </div>
              <div>
                <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>Direct Line Assistance</h2>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>For urgent issues requiring immediate resolution</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {/* Contact Numbers */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Contact Numbers</h3>
                <div className={`p-4 rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className="font-medium mb-1">United States</p>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>+1 (800) 555-0199</p>
                  <p className="font-medium mb-1">International</p>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>+1 (617) 555-0182</p>
                </div>
              </div>

              {/* Hours */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Hours of Operation</h3>
                <div className={`p-4 rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                  <p className={`mb-2 flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiClock className="mr-2" />
                    <span className="font-medium">Weekdays:</span> 8:00 AM - 8:00 PM EST
                  </p>
                  <p className={`flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiClock className="mr-2" />
                    <span className="font-medium">Weekends:</span> 10:00 AM - 4:00 PM EST
                  </p>
                </div>
              </div>
            </div>

            {/* Notes */}
            <div className={`p-4 rounded-lg border ${isDarkMode ? "bg-yellow-900 border-yellow-700" : "bg-yellow-50 border-yellow-100"}`}>
              <h3 className={`text-lg font-medium mb-2 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiGlobe className="mr-2 text-yellow-600" />
                Important Notes
              </h3>
              <ul className={`list-disc pl-5 space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Phone support is for Enterprise customers only</li>
                <li>Have your customer ID ready (found in Account Settings)</li>
                <li>Average wait time: 7 minutes during peak hours</li>
                <li>For quicker service, use our callback option</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PhoneSupport;