import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiUser, FiMail, FiLock, FiCheckCircle, FiLogIn } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const SettingUpYourAccount = () => {
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
      {/* Sidebar */}
      <div>
        <SideNavbar />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Account Setup</h1>
          
          {/* Registration Section */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiUser className="mr-2 text-blue-600" />
              Create Your Account
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Step-by-Step Guide */}
              <div>
                <ol className="space-y-6">
                  <li className="flex items-start">
                    <div className="flex-shrink-0 bg-blue-100 text-blue-600 rounded-full p-2 mr-4">
                      <FiUser className="text-lg" />
                    </div>
                    <div>
                      <h3 className={`text-lg font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Registration Form</h3>
                      <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>Fill in the required fields:</p>
                      <ul className={`mt-2 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                        {[
                          ["Email:", "Valid email address"],
                          ["Password:", "8+ characters"]
                        ].map(([label, desc], i) => (
                          <li className="flex items-start" key={i}>
                            <FiCheckCircle className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                            <span><span className="font-medium">{label}</span> {desc}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </li>
                </ol>
              </div>

              {/* Visual Demo */}
              <div className={`p-6 rounded-lg border ${isDarkMode ? "bg-gray-700 border-gray-600" : "bg-gray-50 border-gray-200"}`}>
                <div className="mb-4 text-center">
                  <h3 className={`text-lg font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Example Registration</h3>
                  <div className={`inline-block p-4 rounded-lg shadow-xs border ${isDarkMode ? "bg-gray-800 border-gray-600" : "bg-white border-gray-300"}`}>
                    <div className="space-y-4">
                      {["Email", "Password"].map((label, i) => (
                        <div key={i}>
                          <label className={`block text-sm font-medium mb-1 ${isDarkMode ? "text-gray-200" : "text-gray-700"}`}>{label}</label>
                          <div className={`h-9 rounded-md px-2 flex items-center ${isDarkMode ? "bg-gray-600 text-gray-200" : "bg-gray-100"}`}>
                            {label === "Email" ? "help@gmail.com" : "*****"}
                          </div>
                        </div>
                      ))}
                      <button className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors">
                        Register
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Login Section */}
          <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiLogIn className="mr-2 text-purple-600" />
              Existing Users
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Sign In Process</h3>
                <ol className={`space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  {[
                    'Click "Sign in" ',
                    'Enter your registered email and password',
                    'Click "Login" to access the application'
                  ].map((step, i) => (
                    <li key={i} className="flex items-start">
                      <span className="bg-purple-100 text-purple-600 rounded-full w-5 h-5 flex items-center justify-center mr-3 mt-0.5 text-sm">{i + 1}</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SettingUpYourAccount;