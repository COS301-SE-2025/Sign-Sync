import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiMessageSquare, FiUser, FiZap } from 'react-icons/fi';

const LiveChatSupport = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Live Chat Support</h1>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <div className="flex items-start mb-6">
              <div className="bg-green-100 p-3 rounded-full mr-4">
                <FiMessageSquare className="text-green-600 text-xl" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Real-Time Assistance</h2>
                <p className="text-gray-600">Get immediate help from our support team</p>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Availability</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-600 mb-2">
                    <span className="font-medium">Monday-Friday:</span> 9:00 AM - 5:00 PM EST
                  </p>
                  <p className="text-gray-600">
                    <span className="font-medium">Saturday-Sunday:</span> Closed
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">How to Access</h3>
                <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                  <li>Click the chat icon in the bottom-right corner</li>
                  <li>Sign in with your account</li>
                  <li>Describe your issue to the bot</li>
                  <li>Wait for a human agent to join</li>
                </ol>
              </div>

              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                <h3 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                  <FiZap className="mr-2 text-blue-600" />
                  Pro Tips
                </h3>
                <ul className="list-disc pl-5 space-y-1 text-gray-600">
                  <li>Have your account email ready</li>
                  <li>Chat sessions timeout after 15 minutes of inactivity</li>
                  <li>You'll receive a transcript via email</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default LiveChatSupport;