import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiPieChart, FiActivity, FiCalendar, FiStar } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const DashboardGuide = () => {
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
            Dashboard Guide
          </h1>

          <div className={`p-6 rounded-xl border mb-8 
            ${isDarkMode ? "bg-purple-950 border-purple-800" : "bg-purple-50 border-purple-100"}`}>
            <h2 className="text-xl font-semibold text-purple-500 mb-2">Your Control Center</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Understand and navigate your SignSync dashboard
            </p>
          </div>

          <div className="space-y-6">
            {/* Overview Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiPieChart className="mr-2 text-blue-500" />
                Overview Panel
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  { title: "Usage Statistics", desc: "View your weekly/monthly translation activity" },
                  { title: "Recent Activity", desc: "Quick access to your last 5 translations" },
                  { title: "System Status", desc: "Check for any service interruptions" }
                ].map((card, i) => (
                  <div key={i} className={`p-4 rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                    <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>{card.title}</h3>
                    <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{card.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* History Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiActivity className="mr-2 text-green-500" />
                Translation History
              </h2>
              <div className="space-y-4">
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  Access your complete translation history with search and filter options:
                </p>
                <ul className={`list-disc pl-5 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  <li>Search by date range or keywords</li>
                  <li>Filter by translation mode (sign/speech)</li>
                  <li>Export history as CSV or PDF</li>
                  <li>Pin frequently used translations</li>
                </ul>
              </div>
            </div>

            {/* Customization Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiStar className="mr-2 text-yellow-500" />
                Customization Options
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Layout Preferences</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Choose between grid or list view</li>
                    <li>• Adjust card sizes</li>
                    <li>• Show/hide specific widgets</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Quick Actions</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Create custom shortcuts</li>
                    <li>• Set up workflow automations</li>
                    <li>• Configure notification preferences</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DashboardGuide;