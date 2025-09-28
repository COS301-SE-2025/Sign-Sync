import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiCamera, FiMic, FiUser, FiSettings, FiHelpCircle, FiLogIn } from 'react-icons/fi';
import PreferenceManager from '../../components/PreferenceManager';

const FirstStepsTutorial = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    // <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
    <section
      className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
      style={{
        background: isDarkMode
          ? "linear-gradient(135deg, #080C1A, #172034)"
          : "#f5f5f5",
      }}
    >
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            First Steps Tutorial
          </h1>

          {/* Welcome Section */}
          <div className={`p-6 rounded-xl border mb-8 ${isDarkMode ? "bg-blue-950 border-blue-800" : "bg-blue-50 border-blue-100"}`}>
            <h2 className="text-2xl font-semibold text-blue-500 mb-2">Welcome to SignSync!</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Follow this guide to start using our sign language translation platform.
            </p>
          </div>

          {/* Account Setup */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiUser className="mr-2 text-blue-600" />
              1. Account Setup
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                {
                  title: "New Users",
                  steps: [
                    `Click "Register" on the main page`,
                    <>
                      Fill in your details:
                      <ul className="list-disc pl-5 mt-1">
                        <li>Email (e.g., apolloprojects.cos301@gmail.com)</li>
                        <li>Password</li>
                      </ul>
                    </>,
                    `Click the "Register" button`,
                  ]
                },
                {
                  title: "Existing Users",
                  steps: [
                    `Click "Sign in"`,
                    <>
                      Enter your credentials:
                      <ul className="list-disc pl-5 mt-1">
                        <li>Email</li>
                        <li>Password</li>
                      </ul>
                    </>,
                    `Click the "Login" button`
                  ]
                }
              ].map(({ title, steps }, idx) => (
                <div key={idx} className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <h3 className={`text-lg font-medium mb-3 ${isDarkMode ? "text-white" : "text-gray-800"}`}>{title}</h3>
                  <ol className={`list-decimal pl-5 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    {steps.map((step, i) => <li key={i}>{step}</li>)}
                  </ol>
                </div>
              ))}
            </div>
          </div>

          {/* Settings */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiSettings className="mr-2 text-purple-600" />
              2. Personalize Your Settings
            </h2>
            <div className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
              <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Go to <span className="font-semibold">Settings</span> (gear icon or tab)</li>
                <li>
                  Configure your preferences:
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
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiCamera className="mr-2 text-green-600" />
              3. Using the Translator
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                {
                  icon: <FiCamera className="mr-2" />,
                  label: "Sign-to-Speech Mode",
                  steps: [
                    `Select the "Sign" button`,
                    `Allow camera access when prompted`,
                    <>
                      Position your hands in view:
                      <ul className="list-disc pl-5 mt-1">
                        <li>System will show detection status</li>
                        <li>"No hand detected" disappears when successful</li>
                      </ul>
                    </>,
                    `Perform signs naturally`
                  ]
                },
                {
                  icon: <FiMic className="mr-2" />,
                  label: "Speech-to-Sign Mode",
                  steps: [
                    `Select the "Speech" button`,
                    `Allow microphone access`,
                    `Speak clearly into your microphone`,
                    `View the avatar's sign language translation`
                  ]
                }
              ].map(({ icon, label, steps }, i) => (
                <div key={i} className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <h3 className={`text-lg font-medium mb-3 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    {icon} {label}
                  </h3>
                  <ol className={`list-decimal pl-5 space-y-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    {steps.map((step, j) => <li key={j}>{step}</li>)}
                  </ol>
                </div>
              ))}
            </div>
          </div>

          {/* Learning Resources */}
          <div className={`p-6 rounded-xl shadow-sm border mb-8 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              <FiHelpCircle className="mr-2 text-yellow-600" />
              4. Learning Resources
            </h2>
            <div className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
              <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>
                  Visit the <span className="font-semibold">"Education"</span> tab for:
                  <ul className="list-disc pl-5 mt-1">
                    <li>Sign language tutorials</li>
                    <li>Practice exercises</li>
                    <li>Learning progress tracking</li>
                  </ul>
                </li>
                <li>
                  Check the <span className="font-semibold">"Help Menu"</span> for:
                  <ul className="list-disc pl-5 mt-1">
                    <li>Video guides ("Tutorials & Videos")</li>
                    <li>Troubleshooting ("FAQs")</li>
                    <li>Live support options</li>
                  </ul>
                </li>
              </ol>
            </div>
          </div>

          {/* Troubleshooting Table */}
          <div className={`p-6 rounded-xl shadow-sm border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <h2 className={`text-2xl font-semibold mb-4 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              Quick Troubleshooting
            </h2>
            <div className="overflow-x-auto">
              <table className={`min-w-full rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                <thead>
                  <tr className={`${isDarkMode ? "border-b border-gray-600" : "border-b border-gray-200"}`}>
                    <th className={`px-4 py-2 text-left ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>Issue</th>
                    <th className={`px-4 py-2 text-left ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>Solution</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["Camera not working", "Check browser permissions"],
                    ["Poor hand detection", "Improve lighting conditions"],
                    ["Translation errors", "Check Help Menu > Troubleshooting"],
                  ].map(([issue, solution], i) => (
                    <tr key={i} className={`${i !== 3 ? (isDarkMode ? "border-b border-gray-600" : "border-b border-gray-200") : ""}`}>
                      <td className={`px-4 py-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{issue}</td>
                      <td className={`px-4 py-2 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{solution}</td>
                    </tr>
                  ))}
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