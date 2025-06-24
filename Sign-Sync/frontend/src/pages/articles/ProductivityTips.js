import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiZap, FiClock, FiBookmark, FiCommand } from 'react-icons/fi';

const ProductivityTips = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Productivity Tips</h1>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiZap className="mr-2 text-yellow-600" />
              Work Faster
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-800 mb-2 flex items-center">
                  <FiClock className="mr-2 text-blue-500" />
                  Time Savers
                </h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• Create custom sign shortcuts</li>
                  <li>• Set up frequent phrases</li>
                  <li>• Use template responses</li>
                </ul>
              </div>

              <div>
                <h3 className="font-medium text-gray-800 mb-2 flex items-center">
                  <FiCommand className="mr-2 text-purple-500" />
                  Keyboard Shortcuts
                </h3>
                <ul className="space-y-2 text-gray-600">
                  <li>• Ctrl+S: Save current translation</li>
                  <li>• Ctrl+F: Quick search</li>
                  <li>• Ctrl+R: Repeat last action</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiBookmark className="mr-2 text-green-600" />
              Organization
            </h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-gray-800 mb-1">Categories</h3>
                <p className="text-gray-600 text-sm">
                  Organize your saved translations into custom categories
                </p>
              </div>

              <div>
                <h3 className="font-medium text-gray-800 mb-1">Tags</h3>
                <p className="text-gray-600 text-sm">
                  Add tags to quickly find related translations
                </p>
              </div>

              <div>
                <h3 className="font-medium text-gray-800 mb-1">Favorites</h3>
                <p className="text-gray-600 text-sm">
                  Star frequently used items for quick access
                </p>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">Want More Tips?</h2>
            <p className="text-gray-600 mb-4">Join our productivity webinar every Thursday.</p>
            <a href="/webinars" className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
              Register Now
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProductivityTips;