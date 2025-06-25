import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiDownload, FiDatabase, FiLock, FiFileText } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const DataExport = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Data Export Guide
          </h1>

          <div className={`p-6 rounded-xl border mb-8 
            ${isDarkMode ? "bg-green-950 border-green-800" : "bg-green-50 border-green-100"}`}>
            <h2 className="text-xl font-semibold text-green-500 mb-2">Export Your Information</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Download your translation history and account data
            </p>
          </div>

          <div className="space-y-6">
            {/* Export Options */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiDownload className="mr-2 text-blue-500" />
                Export Options
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  {
                    title: "CSV Export",
                    description: "Spreadsheet format for data analysis",
                    icon: <FiFileText className="text-blue-600" />,
                    bg: "bg-blue-100"
                  },
                  {
                    title: "JSON Export",
                    description: "Structured data for developers",
                    icon: <FiDatabase className="text-purple-600" />,
                    bg: "bg-purple-100"
                  },
                  {
                    title: "PDF Report",
                    description: "Printable document with your history",
                    icon: <FiFileText className="text-green-600" />,
                    bg: "bg-green-100"
                  }
                ].map(({ title, description, icon, bg }, i) => (
                  <div key={i} className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                    <div className={`${bg} p-2 rounded-full w-10 h-10 flex items-center justify-center mb-2`}>
                      {icon}
                    </div>
                    <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>{title}</h3>
                    <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{description}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Step-by-Step Guide */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                How to Export Your Data
              </h2>
              <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li><span className="font-medium">Navigate to Account Settings</span> → Data Management</li>
                <li><span className="font-medium">Select Export Type:</span> Choose between full account data or specific date range</li>
                <li><span className="font-medium">Choose Format:</span> CSV, JSON, or PDF</li>
                <li><span className="font-medium">Initiate Export:</span> Large exports may take several minutes</li>
                <li><span className="font-medium">Download:</span> You'll receive an email with download link when ready</li>
              </ol>
            </div>

            {/* Security Note */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiLock className="mr-2 text-red-500" />
                Data Security
              </h2>
              <div className={`space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <p><span className="font-medium">Encryption:</span> All exports are encrypted with AES-256</p>
                <p><span className="font-medium">Access:</span> Download links expire after 24 hours</p>
                <p><span className="font-medium">Sensitive Data:</span> API keys are never included in exports</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DataExport;