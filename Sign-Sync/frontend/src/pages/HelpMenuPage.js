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
                        <h1 className="text-3xl font-bold text-gray-800 mb-6">Help Center</h1>
                        
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
                                    <li><a href="#" className="text-blue-600 hover:underline">Product overview</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">First steps tutorial</a></li>
                                    <li><a href="#" className="text-blue-600 hover:underline">Setting up your account</a></li>
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

                           
                        </div>

                       
                    </div>
                </div>
            </section>
        );
    }
}

export default HelpMenuPage;