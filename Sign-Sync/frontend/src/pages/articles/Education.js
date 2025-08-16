import React from "react";
import SideNavbar from "../../components/sideNavbar";
import { FaTrophy, FaBook, FaRunning } from "react-icons/fa";
import PreferenceManager from '../../components/PreferenceManager';

const Education = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section
      className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
      style={{
        background: isDarkMode
          ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
          : "linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)",
      }}
    >
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Education Tab Guide
          </h1>

          {/* Achievements */}
          <div className={`p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border mb-8 
            ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-center mb-4">
              <div className="bg-yellow-100 p-3 rounded-full mr-4">
                <FaTrophy className="text-yellow-600 text-xl" />
              </div>
              <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>Achievements</h2>
            </div>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>
              Track your learning progress and earned badges.
            </p>
            <ul className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} space-y-2`}>
              <li><span className="font-medium">Progress Tracking:</span> See completion percentages for each category</li>
              <li><span className="font-medium">Badges:</span> Earn rewards for milestones</li>
              <li><span className="font-medium">Streaks:</span> Maintain daily learning streaks</li>
            </ul>
          </div>

          {/* Learn */}
          <div className={`p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border mb-8 
            ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-center mb-4">
              <div className="bg-blue-100 p-3 rounded-full mr-4">
                <FaBook className="text-blue-600 text-xl" />
              </div>
              <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>Learn</h2>
            </div>

            <div className="mb-4">
              <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Alphabet</h3>
              <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-2`}>
                Master sign language letters with interactive lessons.
              </p>
              <ul className={`list-disc pl-5 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Video demonstrations for each letter</li>
                <li>Slow-motion replays</li>
                <li>Practice exercises</li>
              </ul>
            </div>

            <div>
              <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Words</h3>
              <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-2`}>
                Learn vocabulary organized by categories.
              </p>
              <ul className={`list-disc pl-5 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Common phrases and greetings</li>
                <li>Themed vocabulary (food, family, etc.)</li>
                <li>Record and compare feature</li>
              </ul>
            </div>
          </div>

          {/* Practice */}
          <div className={`p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border mb-8 
            ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex items-center mb-4">
              <div className="bg-green-100 p-3 rounded-full mr-4">
                <FaRunning className="text-green-600 text-xl" />
              </div>
              <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>Practice</h2>
            </div>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>
              Test your knowledge with interactive exercises.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {["Flashcards", "Multiple Choice", "Freeform"].map((mode, idx) => (
                <div key={idx} className={`${isDarkMode ? "bg-gray-700" : "bg-gray-50"} p-4 rounded-lg`}>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>{mode}</h3>
                  <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>
                    {mode === "Flashcards" && "Sign the displayed word or phrase"}
                    {mode === "Multiple Choice" && "Match signs to their meanings"}
                    {mode === "Freeform" && "Receive feedback on your signing"}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Help Section */}
          <div className={`p-6 rounded-lg border 
            ${isDarkMode ? "bg-blue-950 border-blue-800" : "bg-blue-50 border-blue-100"}`}>
            <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
              Need Help With Education Features?
            </h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>
              Our support team can answer any questions about learning tools.
            </p>
            <button className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
              Contact Support
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Education;