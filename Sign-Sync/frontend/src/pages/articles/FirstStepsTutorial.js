import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiCamera, FiMic, FiUser, FiSettings, FiHelpCircle, FiLogIn } from 'react-icons/fi';

const FirstStepsTutorial = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <div>
        <SideNavbar />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">First Steps Tutorial</h1>
          
          {/* Welcome Section */}
          <div className="bg-blue-50 p-6 rounded-xl border border-blue-100 mb-8">
            <h2 className="text-2xl font-semibold text-blue-800 mb-2">Welcome to SignSync!</h2>
            <p className="text-gray-700">
              Follow this guide to start using our sign language translation platform.
            </p>
          </div>

          {/* Account Setup */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiUser className="mr-2 text-blue-600" />
              1. Account Setup
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Registration */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-gray-800 mb-3">New Users</h3>
                <ol className="list-decimal pl-5 space-y-2 text-gray-600">
                  <li>Click <span className="font-semibold">"Register"</span> on the main page</li>
                  <li>Fill in your details:
                    <ul className="list-disc pl-5 mt-1">
                      <li>Username</li>
                      <li>Email (e.g., apolloprojects.cos301@gmail.com)</li>
                      <li>Password</li>
                    </ul>
                  </li>
                  <li>Click the <span className="font-semibold">"Register"</span> button</li>
                  <li>Verify your email if required</li>
                </ol>
              </div>

              {/* Login */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-gray-800 mb-3">Existing Users</h3>
                <ol className="list-decimal pl-5 space-y-2 text-gray-600">
                  <li>Click <span className="font-semibold">"Sign in"</span></li>
                  <li>Enter your credentials:
                    <ul className="list-disc pl-5 mt-1">
                      <li>Username/Email</li>
                      <li>Password</li>
                    </ul>
                  </li>
                  <li>Click the <span className="font-semibold">"Login"</span> button</li>
                </ol>
              </div>
            </div>
          </div>

          {/* Initial Configuration */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiSettings className="mr-2 text-purple-600" />
              2. Personalize Your Settings
            </h2>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                <li>Go to <span className="font-semibold">Settings</span> (gear icon or tab)</li>
                <li>Configure your preferences:
                  <ul className="list-disc pl-5 mt-1 space-y-1">
                    <li><span className="font-medium">Display Mode:</span> Light/Dark</li>
                    <li><span className="font-medium">Preferred Avatar:</span> Choose style and size</li>
                    <li><span className="font-medium">Font Size:</span> Adjust for readability</li>
                    <li><span className="font-medium">Performance:</span> Toggle between Speed/Accuracy</li>
                  </ul>
                </li>
                <li>Click <span className="font-semibold">"Save"</span> to apply changes</li>
              </ol>
            </div>
          </div>

          {/* Using the Translator */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiCamera className="mr-2 text-green-600" />
              3. Using the Translator
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Sign-to-Speech */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-gray-800 mb-3 flex items-center">
                  <FiCamera className="mr-2" />
                  Sign-to-Speech Mode
                </h3>
                <ol className="list-decimal pl-5 space-y-2 text-gray-600">
                  <li>Select the <span className="font-semibold">"Sign"</span> button</li>
                  <li>Allow camera access when prompted</li>
                  <li>Position your hands in view:
                    <ul className="list-disc pl-5 mt-1">
                      <li>System will show detection status</li>
                      <li>"No hand detected" disappears when successful</li>
                    </ul>
                  </li>
                  <li>Perform signs naturally</li>
                </ol>
              </div>

              {/* Speech-to-Sign */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-gray-800 mb-3 flex items-center">
                  <FiMic className="mr-2" />
                  Speech-to-Sign Mode
                </h3>
                <ol className="list-decimal pl-5 space-y-2 text-gray-600">
                  <li>Select the <span className="font-semibold">"Speech"</span> button</li>
                  <li>Allow microphone access</li>
                  <li>Speak clearly into your microphone</li>
                  <li>View the avatar's sign language translation</li>
                </ol>
              </div>
            </div>
          </div>

          {/* Learning Resources */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiHelpCircle className="mr-2 text-yellow-600" />
              4. Learning Resources
            </h2>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                <li>Visit the <span className="font-semibold">"Education"</span> tab for:
                  <ul className="list-disc pl-5 mt-1">
                    <li>Sign language tutorials</li>
                    <li>Practice exercises</li>
                    <li>Learning progress tracking</li>
                  </ul>
                </li>
                <li>Check the <span className="font-semibold">"Help Menu"</span> for:
                  <ul className="list-disc pl-5 mt-1">
                    <li>Video guides ("Tutorials & Videos")</li>
                    <li>Troubleshooting ("FAQs")</li>
                    <li>Live support options</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>

          {/* Troubleshooting */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Quick Troubleshooting</h2>
            
            <div className="overflow-x-auto">
              <table className="min-w-full bg-gray-50 rounded-lg">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="px-4 py-2 text-left text-gray-700">Issue</th>
                    <th className="px-4 py-2 text-left text-gray-700">Solution</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-200">
                    <td className="px-4 py-2 text-gray-600">Camera not working</td>
                    <td className="px-4 py-2 text-gray-600">Check browser permissions</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="px-4 py-2 text-gray-600">Poor hand detection</td>
                    <td className="px-4 py-2 text-gray-600">Improve lighting conditions</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="px-4 py-2 text-gray-600">Translation errors</td>
                    <td className="px-4 py-2 text-gray-600">Check Help Menu {'>'} Troubleshooting</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 text-gray-600">Performance lag</td>
                    <td className="px-4 py-2 text-gray-600">Adjust Settings {'>'} Performance mode</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FirstStepsTutorial;