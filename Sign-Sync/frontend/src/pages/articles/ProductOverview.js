import React from "react";
import SideNavbar from "../../components/sideNavbar";
import { FiGlobe, FiUser, FiActivity, FiLock, FiSmile, FiVideo } from "react-icons/fi";
import PreferenceManager from "../../components/PreferenceManager";
import logo from "../../assets/Apollo_Projects_Logo.png"

const ProductOverview = () => {
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      {/* Sidebar */}
      <div>
        <SideNavbar />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-5xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className={`text-4xl font-bold mb-4 ${isDarkMode ? "text-white" : "text-gray-900"}`}>
              SignSync Communication Platform
            </h1>
            <p className={`text-xl max-w-3xl mx-auto ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
              Bridging communication gaps between spoken and sign language through AI-powered real-time translation
            </p>
          </div>

          {/* Value Proposition Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {[{
              icon: <FiGlobe className="text-blue-600 text-xl" />,
              title: "Inclusive Communication",
              bg: "bg-blue-100",
              desc: "Break down barriers with seamless two-way translation between speech and sign language"
            }, {
              icon: <FiActivity className="text-purple-600 text-xl" />,
              title: "AI-Powered Accuracy",
              bg: "bg-purple-100",
              desc: "Leveraging cutting-edge machine learning for precise translations that improve with use"
            }, {
              icon: <FiUser className="text-green-600 text-xl" />,
              title: "User-Centric Design",
              bg: "bg-green-100",
              desc: "Intuitive interface with customizable accessibility features for all users"
            }].map(({ icon, title, bg, desc }, i) => (
              <div
                key={i}
                className={`p-6 rounded-xl shadow-sm border transition-all hover:shadow-md ${
                  isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"
                }`}
              >
                <div className={`${bg} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
                  {icon}
                </div>
                <h3 className={`text-lg font-semibold mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>{title}</h3>
                <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-sm`}>{desc}</p>
              </div>
            ))}
          </div>

          {/* Product Showcase */}
          <div className={`p-8 rounded-xl shadow-sm border mb-12 ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="md:w-1/2">
                <h2 className={`text-2xl font-semibold mb-4 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                  Revolutionizing Communication
                </h2>
                <p className={`mb-6 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  SignSync transforms how hearing and deaf communities interact by providing instantaneous,
                  accurate translations between spoken language and sign language.
                </p>
                {[{
                  icon: <FiVideo className="text-blue-500" />,
                  title: "Real-Time Translation",
                  desc: "Less than 500ms latency for natural conversations",
                  bg: "bg-blue-50"
                }, {
                  icon: <FiSmile className="text-green-500" />,
                  title: "Accessibility First",
                  desc: "WCAG 2.1 AA compliant interface",
                  bg: "bg-green-50"
                }].map(({ icon, title, desc, bg }, i) => (
                  <div key={i} className="flex items-start mb-4">
                    <div className={`${bg} p-2 rounded-lg mr-4`}>{icon}</div>
                    <div>
                      <h4 className={`font-medium ${isDarkMode ? "text-white" : "text-gray-800"}`}>{title}</h4>
                      <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className={`md:w-1/2 rounded-lg flex items-center justify-center ${isDarkMode ? "bg-gray-700" : "bg-gray-100"}`}>
                <img 
                  src={logo} 
                  alt="SignSync Web Application Preview" 
                  className="max-w-full max-h-full object-contain p-6" 
                />
              </div>
            </div>
          </div>

          {/* Technical Specifications */}
          <div className="mb-12">
            <h2 className={`text-2xl font-semibold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Technical Specifications</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[{
                title: "Core Capabilities",
                icon: <FiActivity />,
                iconBg: "bg-blue-100 text-blue-600",
                list: [
                  "Speech-to-Sign: 95% accuracy for English (general)",
                  "Sign-to-Text: 90% accuracy for ASL",
                  "Adaptive learning model improves with usage"
                ],
                dotColor: "text-blue-500"
              }, {
                title: "System Requirements",
                icon: <FiGlobe />,
                iconBg: "bg-purple-100 text-purple-600",
                list: [
                  "Web: Chrome, Firefox, Edge (latest versions)",
                  "Camera: 720p minimum resolution",
                  "Internet: 5Mbps stable connection"
                ],
                dotColor: "text-purple-500"
              }].map(({ title, icon, iconBg, list, dotColor }, i) => (
                <div key={i} className={`p-6 rounded-xl border ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
                  <h3 className={`text-lg font-medium mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                    <span className={`${iconBg} p-2 rounded-lg mr-3`}>{icon}</span>
                    {title}
                  </h3>
                  <ul className="space-y-3">
                    {list.map((item, j) => (
                      <li key={j} className="flex items-start">
                        <span className={`${dotColor} mr-2`}>•</span>
                        <span className={`${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          
        </div>
      </div>
    </section>
  );
};

export default ProductOverview;