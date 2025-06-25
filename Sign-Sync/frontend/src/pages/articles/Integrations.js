import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiLink, FiSlack, FiChrome, FiCode } from 'react-icons/fi';

const Integrations = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Integrations</h1>
          
          <div className="bg-purple-50 p-6 rounded-xl border border-purple-100 mb-8">
            <h2 className="text-xl font-semibold text-purple-800 mb-2">Connect with Your Favorite Tools</h2>
            <p className="text-gray-700">Extend SignSync's functionality with these integrations</p>
          </div>

          <div className="space-y-6">
            {/* Available Integrations */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiLink className="mr-2 text-blue-500" />
                Available Integrations
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-2">
                    <FiSlack className="text-purple-500 mr-2" />
                    <h3 className="font-medium text-gray-800">Slack</h3>
                  </div>
                  <p className="text-gray-600 text-sm">
                    Get translations directly in your Slack channels
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-2">
                    <FiChrome className="text-green-500 mr-2" />
                    <h3 className="font-medium text-gray-800">Chrome Extension</h3>
                  </div>
                  <p className="text-gray-600 text-sm">
                    Translate web content with one click
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-2">
                    <FiCode className="text-orange-500 mr-2" />
                    <h3 className="font-medium text-gray-800">Zapier</h3>
                  </div>
                  <p className="text-gray-600 text-sm">
                    Connect with 2000+ apps through Zapier
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="flex items-center mb-2">
                    <FiLink className="text-red-500 mr-2" />
                    <h3 className="font-medium text-gray-800">Microsoft Teams</h3>
                  </div>
                  <p className="text-gray-600 text-sm">
                    Real-time translation during meetings
                  </p>
                </div>
              </div>
            </div>

            {/* Setup Guide */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Setup Instructions</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Webhook Integration</h3>
                  <div className="bg-gray-800 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                    {`// Sample webhook payload\n`}
                    {`{\n`}
                    {`  "event": "translation_complete",\n`}
                    {`  "data": {\n`}
                    {`    "input": "Hello world",\n`}
                    {`    "output": "[Sign language video URL]",\n`}
                    {`    "timestamp": "2023-07-15T12:00:00Z"\n`}
                    {`  }\n`}
                    {`}`}
                  </div>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">OAuth Configuration</h3>
                  <p className="text-gray-600 text-sm">
                    Most integrations use OAuth 2.0 for secure connection. You'll need to:
                  </p>
                  <ol className="list-decimal pl-5 mt-2 space-y-1 text-gray-600 text-sm">
                    <li>Authorize SignSync in the target application</li>
                    <li>Configure permissions in our integration settings</li>
                    <li>Test the connection</li>
                  </ol>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Integrations;