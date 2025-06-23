import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiUser, FiEye, FiMoon, FiAward, FiGlobe } from 'react-icons/fi';

const AccountSettings = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Account Settings Guide</h1>
          
          <div className="space-y-6">
            {/* Profile Settings */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiUser className="mr-2 text-blue-600" />
                Profile Configuration
              </h2>
              <ul className="space-y-4 text-gray-600">
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-600 rounded-full p-1 mr-3">
                    <FiUser className="text-sm" />
                  </span>
                  <div>
                    <h3 className="font-medium text-gray-800">Update Personal Information</h3>
                    <p className="text-sm">Change your display name, email, or profile picture</p>
                  </div>
                </li>
                <li className="flex items-start">
                  <span className="bg-blue-100 text-blue-600 rounded-full p-1 mr-3">
                    <FiGlobe className="text-sm" />
                  </span>
                  <div>
                    <h3 className="font-medium text-gray-800">Language Preferences</h3>
                    <p className="text-sm">Set your preferred language for the interface</p>
                  </div>
                </li>
              </ul>
            </div>

            {/* Accessibility */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiEye className="mr-2 text-purple-600" />
                Accessibility Settings
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Display Options</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Light/Dark mode toggle</li>
                    <li>• Font size adjustment</li>
                    <li>• High contrast mode</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Avatar Customization</h3>
                  <ul className="text-sm text-gray-600 space-y-1">
                    <li>• Change avatar style</li>
                    <li>• Adjust animation speed</li>
                    <li>• Enable/disable gestures</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Advanced Settings */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiAward className="mr-2 text-green-600" />
                Advanced Options
              </h2>
              <div className="space-y-3 text-gray-600">
                <p><span className="font-medium">Data Export:</span> Download your translation history</p>
                <p><span className="font-medium">Account Deletion:</span> Permanently remove your account</p>
                <p><span className="font-medium">API Access:</span> Connect third-party applications</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AccountSettings;