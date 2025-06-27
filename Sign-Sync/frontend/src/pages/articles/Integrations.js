import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiLink, FiSlack, FiChrome, FiCode } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const Integrations = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Integrations</h1>

          {/* Intro Section */}
          <div className={`p-6 rounded-xl border mb-8 ${isDarkMode ? "bg-purple-950 border-purple-800" : "bg-purple-50 border-purple-100"}`}>
            <h2 className="text-xl font-semibold text-purple-500 mb-2">Connect with Your Favorite Tools</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Extend SignSync's functionality with these integrations
            </p>
          </div>

          {/* Available Integrations */}
          <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiLink className="mr-2 text-blue-500" />
              Available Integrations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { icon: <FiSlack className="text-purple-500 mr-2" />, title: "Slack", desc: "Get translations directly in your Slack channels" },
                { icon: <FiChrome className="text-green-500 mr-2" />, title: "Chrome Extension", desc: "Translate web content with one click" },
                { icon: <FiCode className="text-orange-500 mr-2" />, title: "Zapier", desc: "Connect with 2000+ apps through Zapier" },
                { icon: <FiLink className="text-red-500 mr-2" />, title: "Microsoft Teams", desc: "Real-time translation during meetings" },
              ].map((item, idx) => (
                <div key={idx} className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <div className="flex items-center mb-2">
                    {item.icon}
                    <h3 className={`font-medium ${isDarkMode ? "text-white" : "text-gray-800"}`}>{item.title}</h3>
                  </div>
                  <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{item.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Setup Instructions */}
          <div className={`p-6 rounded-xl shadow-sm border mt-6 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-xl font-semibold mb-4 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Setup Instructions</h2>
            <div className="space-y-4">
              <div>
                <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Webhook Integration</h3>
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
                <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>OAuth Configuration</h3>
                <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  Most integrations use OAuth 2.0 for secure connection. You'll need to:
                </p>
                <ol className={`list-decimal pl-5 mt-2 space-y-1 text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  <li>Authorize SignSync in the target application</li>
                  <li>Configure permissions in our integration settings</li>
                  <li>Test the connection</li>
                </ol>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default Integrations;