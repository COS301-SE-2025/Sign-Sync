import React from "react";
import SideNavbar from "../components/sideNavbar";
import { FaQuestionCircle, FaBook, FaVideo, FaEnvelope, FaFileAlt, FaLightbulb } from "react-icons/fa";

class HelpMenuPage extends React.Component {
    render() {
        return (
            <section className="flex h-screen overflow-hidden bg-gray-50">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Main Content */}
                <div className="flex-1 overflow-y-auto p-8">
                    <div className="max-w-4xl mx-auto">
                        <h1 className="text-3xl font-bold text-gray-800 mb-6">Help Menu</h1>
                        
                        {/* Search Help */}
                        <div className="mb-8">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search help articles..."
                                    className="w-full p-4 pl-12 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                />
                                <FaQuestionCircle className="absolute left-4 top-4 text-gray-400 text-xl" />
                            </div>
                        </div>

                        {/* Help Categories */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                            {/* Getting Started */}
                            <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100">
                                <div className="flex items-center mb-4">
                                    <div className="bg-blue-100 p-3 rounded-full mr-4">
                                        <FaBook className="text-blue-600 text-xl" />
                                    </div>
                                    <h2 className="text-xl font-semibold text-gray-800">Getting Started</h2>
                                </div>
                                <p className="text-gray-600 mb-4">New to our product? Start with these guides.</p>
                                <ul className="space-y-2">
                                    <li><a href="/productOverview" className="text-blue-600 hover:underline">Product overview</a></li>
                                    <li><a href="/firstStepsTutorial" className="text-blue-600 hover:underline">First steps tutorial</a></li>
                                    <li><a href="/settingUpYourAccount" className="text-blue-600 hover:underline">Setting up your account</a></li>
                                </ul>
                            </div>

                            {/* Tutorials & Videos */}
                            <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100">
                                <div className="flex items-center mb-4">
                                    <div className="bg-purple-100 p-3 rounded-full mr-4">
                                        <FaVideo className="text-purple-600 text-xl" />
                                    </div>
                                    <h2 className="text-xl font-semibold text-gray-800">Tutorials & Videos</h2>
                                </div>
                                <p className="text-gray-600 mb-4">Watch step-by-step video guides.</p>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-blue-600 hover:underline">Basic features walkthrough</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Advanced techniques</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Productivity tips</a></li>
                                </ul>
                            </div>

                            {/* FAQs */}
                            <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100">
                                <div className="flex items-center mb-4">
                                    <div className="bg-green-100 p-3 rounded-full mr-4">
                                        <FaFileAlt className="text-green-600 text-xl" />
                                    </div>
                                    <h2 className="text-xl font-semibold text-gray-800">FAQs</h2>
                                </div>
                                <p className="text-gray-600 mb-4">Find answers to common questions.</p>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-blue-600 hover:underline">Account settings</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Billing questions</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Troubleshooting</a></li>
                                </ul>
                            </div>

                            {/* Contact Support */}
                            <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100">
                                <div className="flex items-center mb-4">
                                    <div className="bg-yellow-100 p-3 rounded-full mr-4">
                                        <FaEnvelope className="text-yellow-600 text-xl" />
                                    </div>
                                    <h2 className="text-xl font-semibold text-gray-800">Contact Support</h2>
                                </div>
                                <p className="text-gray-600 mb-4">Can't find what you need? We're here to help.</p>
                                <ul className="space-y-2">
                                    <li><a href="#" className="text-blue-600 hover:underline">Email support</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Live chat</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Phone support</a></li>
                                </ul>
                            </div>
                        </div>

                        {/* Popular Articles */}
                        <div className="mb-8">
                            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                                <FaLightbulb className="text-yellow-500 mr-2" /> Popular Help Articles
                            </h2>
                            <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
                                {[
                                    "How to reset your password",
                                    "Understanding your dashboard",
                                    "Exporting your data",
                                    "Integrations with other tools",
                                    "Privacy and security settings"
                                ].map((article, index) => (
                                    <div 
                                        key={index} 
                                        className={`p-4 hover:bg-gray-50 ${index !== 0 ? 'border-t border-gray-100' : ''}`}
                                    >
                                        <a href="#" className="flex justify-between items-center text-gray-800 hover:text-blue-600">
                                            <span>{article}</span>
                                            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                                            </svg>
                                        </a>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Still Need Help? */}
                        <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                            <h2 className="text-xl font-semibold text-gray-800 mb-2">Still need help?</h2>
                            <p className="text-gray-600 mb-4">Our support team is available 24/7 to assist you.</p>
                            <button className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                                Contact Support
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}

export default HelpMenuPage;