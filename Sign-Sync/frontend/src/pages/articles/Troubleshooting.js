import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiCamera, FiMic, FiWifi, FiAlertTriangle, FiMessageSquare } from 'react-icons/fi';

const Troubleshooting = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Troubleshooting Guide</h1>
          
          <div className="space-y-6">
            {/* Connection Issues */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiWifi className="mr-2 text-blue-600" />
                Connection Problems
              </h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Symptoms</h3>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• "No internet connection" error</li>
                    <li>• Frequent disconnections</li>
                    <li>• Slow translation speeds</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Solutions</h3>
                  <ol className="text-gray-600 text-sm space-y-2">
                    <li>1. Check your network connection</li>
                    <li>2. Restart your router</li>
                    <li>3. Try switching between WiFi/mobile data</li>
                    <li>4. Contact your ISP if problems persist</li>
                  </ol>
                </div>
              </div>
            </div>

            {/* Device Compatibility */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiAlertTriangle className="mr-2 text-yellow-600" />
                Device Issues
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2 flex items-center">
                    <FiCamera className="mr-2" />
                    Camera Problems
                  </h3>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Ensure camera permissions are granted</li>
                    <li>• Clean camera lens</li>
                    <li>• Test with another app</li>
                    <li>• Update device drivers</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2 flex items-center">
                    <FiMic className="mr-2" />
                    Microphone Issues
                  </h3>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Check microphone permissions</li>
                    <li>• Ensure correct input device is selected</li>
                    <li>• Reduce background noise</li>
                    <li>• Test microphone volume</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Support Contact */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiMessageSquare className="mr-2 text-green-600" />
                Need More Help?
              </h2>
              <div className="space-y-3 text-gray-600">
                <p><span className="font-medium">Email:</span> support@signsync.com (24hr response)</p>
                <p><span className="font-medium">Live Chat:</span> Available Mon-Fri, 9AM-5PM EST</p>
                <p><span className="font-medium">Emergency:</span> +1 (555) 123-4567 (Critical outages only)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Troubleshooting;