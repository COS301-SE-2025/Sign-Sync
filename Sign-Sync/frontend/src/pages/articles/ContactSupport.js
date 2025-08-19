import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiMail, FiMessageSquare, FiPhone, FiClock, FiAlertCircle } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const ContactSupport = () => {
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
            Contact Support
          </h1>

          <div className={`p-6 rounded-xl border mb-8 
            ${isDarkMode ? "bg-blue-950 border-blue-800" : "bg-blue-50 border-blue-100"}`}>
            <h2 className="text-xl font-semibold text-blue-500 mb-2">We're Here to Help</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Choose the support option that works best for you
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Email Support */}
            <div className={`p-6 rounded-xl shadow-sm border hover:shadow-md transition-shadow 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <div className="bg-blue-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiMail className="text-blue-600 text-xl" />
              </div>
              <h2 className={`text-lg font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Email Support</h2>
              <p className={`text-sm mb-4 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                Get help within 24 hours for non-urgent issues
              </p>
              <div className="text-blue-600 font-medium">support@signsync.com</div>
            </div>

            {/* Live Chat */}
            <div className={`p-6 rounded-xl shadow-sm border hover:shadow-md transition-shadow 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <div className="bg-green-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiMessageSquare className="text-green-600 text-xl" />
              </div>
              <h2 className={`text-lg font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Live Chat</h2>
              <p className={`text-sm mb-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                Mon-Fri: 9AM-5PM EST
              </p>
              <p className={`text-sm mb-4 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                Instant connection with our team
              </p>
              <button className="bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-1.5 px-4 rounded-lg transition-colors">
                Start Chat
              </button>
            </div>

            {/* Phone Support */}
            <div className={`p-6 rounded-xl shadow-sm border hover:shadow-md transition-shadow 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <div className="bg-red-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiPhone className="text-red-600 text-xl" />
              </div>
              <h2 className={`text-lg font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Phone Support</h2>
              <p className={`text-sm mb-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                24/7 for critical issues
              </p>
              <p className={`text-sm mb-4 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                Enterprise customers only
              </p>
              <div className="text-red-600 font-medium">+1 (800) 555-0199</div>
            </div>
          </div>

          {/* Before Contacting Support */}
          <div className={`p-6 rounded-xl border 
            ${isDarkMode ? "bg-yellow-950 border-yellow-800" : "bg-yellow-50 border-yellow-100"}`}>
            <h2 className={`text-xl font-semibold mb-3 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiAlertCircle className="mr-2 text-yellow-600" />
              Before Contacting Support
            </h2>
            <ul className={`list-disc pl-5 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
              <li>
                Check our <a href="/troubleshooting" className="text-blue-600 hover:underline">Troubleshooting Guide</a>
              </li>
              <li>Have your account email ready</li>
              <li>Note any error messages you're receiving</li>
              <li>Prepare screenshots if applicable</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSupport;