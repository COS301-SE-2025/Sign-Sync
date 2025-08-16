import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiMail, FiClock, FiAlertCircle } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const EmailSupport = () => {
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
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Email Support
          </h1>

          {/* Contact Info Card */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 
            ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-start mb-6">
              <div className="bg-blue-100 p-3 rounded-full mr-4">
                <FiMail className="text-blue-600 text-xl" />
              </div>
              <div>
                <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                  How to Contact Us
                </h2>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  Our team is ready to help with any questions or issues
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Standard Support */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                  Standard Support
                </h3>
                <div className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <p className={`mb-2 flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiMail className="mr-2 text-blue-500" />
                    <span className="font-medium">Email:</span> support@signsync.com
                  </p>
                  <p className={`mb-2 flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiClock className="mr-2 text-blue-500" />
                    <span className="font-medium">Response Time:</span> 24-48 hours
                  </p>
                  <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                    Include your account email and detailed description of the issue
                  </p>
                </div>
              </div>

              {/* Priority Support */}
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                  Priority Support
                </h3>
                <div className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <p className={`mb-2 flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiMail className="mr-2 text-purple-500" />
                    <span className="font-medium">Email:</span> priority@signsync.com
                  </p>
                  <p className={`mb-2 flex items-center ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <FiClock className="mr-2 text-purple-500" />
                    <span className="font-medium">Response Time:</span> Under 12 hours
                  </p>
                  <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>
                    Available for Enterprise plan customers only
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Before You Email */}
          <div className={`p-6 rounded-xl border 
            ${isDarkMode ? "bg-yellow-950 border-yellow-800" : "bg-yellow-50 border-yellow-100"}`}>
            <h2 className={`text-xl font-semibold mb-3 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiAlertCircle className="mr-2 text-yellow-600" />
              Before You Email
            </h2>
            <ul className={`list-disc pl-5 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
              <li>
                Check our <a href="/troubleshooting" className="text-blue-500 hover:underline">Troubleshooting Guide</a>
              </li>
              <li>Include screenshots if applicable</li>
              <li>Note any error messages verbatim</li>
              <li>Specify your device/browser version</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EmailSupport;