import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiZap, FiClock, FiBookmark, FiCommand } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const ProductivityTips = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-black'}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Productivity Tips</h1>

          {/* Work Faster Section */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'}`}>
            <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
              <FiZap className="mr-2 text-yellow-600" />
              Work Faster
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className={`font-medium mb-2 flex items-center ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                  <FiClock className="mr-2 text-blue-500" />
                  Time Savers
                </h3>
                <ul className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} space-y-2`}>
                  <li>• Create custom sign shortcuts</li>
                  <li>• Set up frequent phrases</li>
                  <li>• Use template responses</li>
                </ul>
              </div>

              <div>
                <h3 className={`font-medium mb-2 flex items-center ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                  <FiCommand className="mr-2 text-purple-500" />
                  Keyboard Shortcuts
                </h3>
                <ul className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} space-y-2`}>
                  <li>• Ctrl+S: Save current translation</li>
                  <li>• Ctrl+F: Quick search</li>
                  <li>• Ctrl+R: Repeat last action</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Organization Section */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'}`}>
            <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
              <FiBookmark className="mr-2 text-green-600" />
              Organization
            </h2>

            <div className="space-y-4">
              <div>
                <h3 className={`font-medium mb-1 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Categories</h3>
                <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
                  Organize your saved translations into custom categories
                </p>
              </div>

              <div>
                <h3 className={`font-medium mb-1 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Tags</h3>
                <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
                  Add tags to quickly find related translations
                </p>
              </div>

              <div>
                <h3 className={`font-medium mb-1 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Favorites</h3>
                <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
                  Star frequently used items for quick access
                </p>
              </div>
            </div>
          </div>

          {/* Webinar CTA */}
          <div className={`p-6 rounded-xl border ${isDarkMode ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-100'}`}>
            <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Want More Tips?</h2>
            <p className={`${isDarkMode ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
              Join our productivity webinar every Thursday.
            </p>
            <a
              href="/webinars"
              className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
            >
              Register Now
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProductivityTips;