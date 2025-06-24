import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiPieChart, FiActivity, FiCalendar, FiStar } from 'react-icons/fi';

const DashboardGuide = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Dashboard Guide</h1>
          
          <div className="bg-purple-50 p-6 rounded-xl border border-purple-100 mb-8">
            <h2 className="text-xl font-semibold text-purple-800 mb-2">Your Control Center</h2>
            <p className="text-gray-700">Understand and navigate your SignSync dashboard</p>
          </div>

          <div className="space-y-6">
            {/* Overview Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiPieChart className="mr-2 text-blue-500" />
                Overview Panel
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium text-gray-800 mb-1">Usage Statistics</h3>
                  <p className="text-gray-600 text-sm">
                    View your weekly/monthly translation activity
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium text-gray-800 mb-1">Recent Activity</h3>
                  <p className="text-gray-600 text-sm">
                    Quick access to your last 5 translations
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-medium text-gray-800 mb-1">System Status</h3>
                  <p className="text-gray-600 text-sm">
                    Check for any service interruptions
                  </p>
                </div>
              </div>
            </div>

            {/* History Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiActivity className="mr-2 text-green-500" />
                Translation History
              </h2>
              <div className="space-y-4">
                <p className="text-gray-600">
                  Access your complete translation history with search and filter options:
                </p>
                <ul className="list-disc pl-5 space-y-2 text-gray-600">
                  <li>Search by date range or keywords</li>
                  <li>Filter by translation mode (sign/speech)</li>
                  <li>Export history as CSV or PDF</li>
                  <li>Pin frequently used translations</li>
                </ul>
              </div>
            </div>

            {/* Customization Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiStar className="mr-2 text-yellow-500" />
                Customization Options
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Layout Preferences</h3>
                  <ul className="text-gray-600 space-y-1 text-sm">
                    <li>• Choose between grid or list view</li>
                    <li>• Adjust card sizes</li>
                    <li>• Show/hide specific widgets</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Quick Actions</h3>
                  <ul className="text-gray-600 space-y-1 text-sm">
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