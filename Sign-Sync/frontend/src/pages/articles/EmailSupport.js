import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiMail, FiClock, FiAlertCircle } from 'react-icons/fi';

const EmailSupport = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Email Support</h1>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <div className="flex items-start mb-6">
              <div className="bg-blue-100 p-3 rounded-full mr-4">
                <FiMail className="text-blue-600 text-xl" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800">How to Contact Us</h2>
                <p className="text-gray-600">Our team is ready to help with any questions or issues</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Standard Support</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-600 mb-2 flex items-center">
                    <FiMail className="mr-2 text-blue-500" />
                    <span className="font-medium">Email:</span> support@signsync.com
                  </p>
                  <p className="text-gray-600 mb-2 flex items-center">
                    <FiClock className="mr-2 text-blue-500" />
                    <span className="font-medium">Response Time:</span> 24-48 hours
                  </p>
                  <p className="text-gray-600 text-sm">
                    Include your account email and detailed description of the issue
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Priority Support</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-600 mb-2 flex items-center">
                    <FiMail className="mr-2 text-purple-500" />
                    <span className="font-medium">Email:</span> priority@signsync.com
                  </p>
                  <p className="text-gray-600 mb-2 flex items-center">
                    <FiClock className="mr-2 text-purple-500" />
                    <span className="font-medium">Response Time:</span> Under 12 hours
                  </p>
                  <p className="text-gray-600 text-sm">
                    Available for Enterprise plan customers only
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 p-6 rounded-xl border border-yellow-100">
            <h2 className="text-xl font-semibold text-gray-800 mb-3 flex items-center">
              <FiAlertCircle className="mr-2 text-yellow-600" />
              Before You Email
            </h2>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Check our <a href="/troubleshooting" className="text-blue-600 hover:underline">Troubleshooting Guide</a></li>
              <li>Include screenshots if applicable</li>
              <li>Note any error messages verbatim</li>
              <li>Specify your device/browser version</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EmailSupport;