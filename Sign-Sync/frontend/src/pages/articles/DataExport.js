import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiDownload, FiDatabase, FiLock, FiFileText } from 'react-icons/fi';

const DataExport = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Data Export Guide</h1>
          
          <div className="bg-green-50 p-6 rounded-xl border border-green-100 mb-8">
            <h2 className="text-xl font-semibold text-green-800 mb-2">Export Your Information</h2>
            <p className="text-gray-700">Download your translation history and account data</p>
          </div>

          <div className="space-y-6">
            {/* Export Options */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiDownload className="mr-2 text-blue-500" />
                Export Options
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="bg-blue-100 p-2 rounded-full w-10 h-10 flex items-center justify-center mb-2">
                    <FiFileText className="text-blue-600" />
                  </div>
                  <h3 className="font-medium text-gray-800 mb-1">CSV Export</h3>
                  <p className="text-gray-600 text-sm">
                    Spreadsheet format for data analysis
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="bg-purple-100 p-2 rounded-full w-10 h-10 flex items-center justify-center mb-2">
                    <FiDatabase className="text-purple-600" />
                  </div>
                  <h3 className="font-medium text-gray-800 mb-1">JSON Export</h3>
                  <p className="text-gray-600 text-sm">
                    Structured data for developers
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="bg-green-100 p-2 rounded-full w-10 h-10 flex items-center justify-center mb-2">
                    <FiFileText className="text-green-600" />
                  </div>
                  <h3 className="font-medium text-gray-800 mb-1">PDF Report</h3>
                  <p className="text-gray-600 text-sm">
                    Printable document with your history
                  </p>
                </div>
              </div>
            </div>

            {/* Step-by-Step Guide */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">How to Export Your Data</h2>
              <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                <li>
                  <span className="font-medium">Navigate to Account Settings</span> → Data Management
                </li>
                <li>
                  <span className="font-medium">Select Export Type:</span> Choose between full account data or specific date range
                </li>
                <li>
                  <span className="font-medium">Choose Format:</span> CSV, JSON, or PDF
                </li>
                <li>
                  <span className="font-medium">Initiate Export:</span> Large exports may take several minutes
                </li>
                <li>
                  <span className="font-medium">Download:</span> You'll receive an email with download link when ready
                </li>
              </ol>
            </div>

            {/* Security Note */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiLock className="mr-2 text-red-500" />
                Data Security
              </h2>
              <div className="space-y-3 text-gray-600">
                <p>
                  <span className="font-medium">Encryption:</span> All exports are encrypted with AES-256
                </p>
                <p>
                  <span className="font-medium">Access:</span> Download links expire after 24 hours
                </p>
                <p>
                  <span className="font-medium">Sensitive Data:</span> API keys are never included in exports
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DataExport;