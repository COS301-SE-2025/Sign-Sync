import React from "react";
import SideNavbar from "../../components/sideNavbar";
import { FiGlobe, FiUser, FiActivity, FiLock, FiSmile, FiVideo } from "react-icons/fi";

const ProductOverview = () => {
    return (
        <section className="flex h-screen overflow-hidden bg-gray-50">
            {/* Sidebar */}
            <div>
                <SideNavbar />
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto p-8">
                <div className="max-w-5xl mx-auto">
                    {/* Hero Section */}
                    <div className="text-center mb-12">
                        <h1 className="text-4xl font-bold text-gray-900 mb-4">SignSync Communication Platform</h1>
                        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                            Bridging communication gaps between spoken and sign language through AI-powered real-time translation
                        </p>
                    </div>

                    {/* Value Proposition Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all">
                            <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                                <FiGlobe className="text-blue-600 text-xl" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">Inclusive Communication</h3>
                            <p className="text-gray-600 text-sm">
                                Break down barriers with seamless two-way translation between speech and sign language
                            </p>
                        </div>

                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all">
                            <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                                <FiActivity className="text-purple-600 text-xl" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">AI-Powered Accuracy</h3>
                            <p className="text-gray-600 text-sm">
                                Leveraging cutting-edge machine learning for precise translations that improve with use
                            </p>
                        </div>

                        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-all">
                            <div className="bg-green-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                                <FiUser className="text-green-600 text-xl" />
                            </div>
                            <h3 className="text-lg font-semibold text-gray-800 mb-2">User-Centric Design</h3>
                            <p className="text-gray-600 text-sm">
                                Intuitive interface with customizable accessibility features for all users
                            </p>
                        </div>
                    </div>

                    {/* Product Showcase */}
                    <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100 mb-12">
                        <div className="flex flex-col md:flex-row gap-8">
                            <div className="md:w-1/2">
                                <h2 className="text-2xl font-semibold text-gray-800 mb-4">Revolutionizing Communication</h2>
                                <p className="text-gray-600 mb-6">
                                    SignSync transforms how hearing and deaf communities interact by providing instantaneous, 
                                    accurate translations between spoken language and sign language. Our solution combines 
                                    advanced speech recognition with motion capture technology to create a seamless 
                                    two-way communication experience.
                                </p>
                                <div className="space-y-4">
                                    <div className="flex items-start">
                                        <div className="bg-blue-50 p-2 rounded-lg mr-4">
                                            <FiVideo className="text-blue-500" />
                                        </div>
                                        <div>
                                            <h4 className="font-medium text-gray-800">Real-Time Translation</h4>
                                            <p className="text-gray-600 text-sm">Less than 500ms latency for natural conversations</p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="bg-purple-50 p-2 rounded-lg mr-4">
                                            <FiLock className="text-purple-500" />
                                        </div>
                                        <div>
                                            <h4 className="font-medium text-gray-800">Enterprise-Grade Security</h4>
                                            <p className="text-gray-600 text-sm">End-to-end encryption for all communications</p>
                                        </div>
                                    </div>
                                    <div className="flex items-start">
                                        <div className="bg-green-50 p-2 rounded-lg mr-4">
                                            <FiSmile className="text-green-500" />
                                        </div>
                                        <div>
                                            <h4 className="font-medium text-gray-800">Accessibility First</h4>
                                            <p className="text-gray-600 text-sm">WCAG 2.1 AA compliant interface</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className="md:w-1/2 bg-gray-100 rounded-lg flex items-center justify-center">
                                <div className="text-center p-8 text-gray-400">
                                    {/* Replace with actual product screenshot or mockup */}
                                    <p className="mb-4">[Product Interface Mockup]</p>
                                    <p className="text-sm">SignSync Web Application</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Technical Specifications */}
                    <div className="mb-12">
                        <h2 className="text-2xl font-semibold text-gray-800 mb-6">Technical Specifications</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-white p-6 rounded-xl border border-gray-100">
                                <h3 className="text-lg font-medium text-gray-800 mb-4 flex items-center">
                                    <span className="bg-blue-100 text-blue-600 p-2 rounded-lg mr-3">
                                        <FiActivity />
                                    </span>
                                    Core Capabilities
                                </h3>
                                <ul className="space-y-3">
                                    <li className="flex items-start">
                                        <span className="text-blue-500 mr-2">•</span>
                                        <span className="text-gray-600">Speech-to-Sign: 95% accuracy for English (general)</span>
                                    </li>
                                    <li className="flex items-start">
                                        <span className="text-blue-500 mr-2">•</span>
                                        <span className="text-gray-600">Sign-to-Text: 90% accuracy for ASL</span>
                                    </li>
                                    <li className="flex items-start">
                                        <span className="text-blue-500 mr-2">•</span>
                                        <span className="text-gray-600">Adaptive learning model improves with usage</span>
                                    </li>
                                </ul>
                            </div>

                            <div className="bg-white p-6 rounded-xl border border-gray-100">
                                <h3 className="text-lg font-medium text-gray-800 mb-4 flex items-center">
                                    <span className="bg-purple-100 text-purple-600 p-2 rounded-lg mr-3">
                                        <FiGlobe />
                                    </span>
                                    System Requirements
                                </h3>
                                <ul className="space-y-3">
                                    <li className="flex items-start">
                                        <span className="text-purple-500 mr-2">•</span>
                                        <span className="text-gray-600">Web: Chrome, Firefox, Edge (latest versions)</span>
                                    </li>
                                    <li className="flex items-start">
                                        <span className="text-purple-500 mr-2">•</span>
                                        <span className="text-gray-600">Camera: 720p minimum resolution</span>
                                    </li>
                                    <li className="flex items-start">
                                        <span className="text-purple-500 mr-2">•</span>
                                        <span className="text-gray-600">Internet: 5Mbps stable connection</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Call to Action */}
                    <div className="bg-gradient-to-r from-blue-600 to-blue-800 rounded-xl p-8 text-center text-white">
                        <h2 className="text-2xl font-semibold mb-3">Ready to Transform Communication?</h2>
                        <p className="mb-6 max-w-2xl mx-auto opacity-90">
                            SignSync is available for educational institutions, healthcare providers, and enterprise customers.
                        </p>
                        <div className="space-x-4">
                            <button className="bg-white text-blue-600 font-medium py-2 px-6 rounded-lg hover:bg-blue-50 transition-colors">
                                Request Demo
                            </button>
                            <button className="border border-white text-white font-medium py-2 px-6 rounded-lg hover:bg-white hover:bg-opacity-10 transition-colors">
                                Contact Sales
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default ProductOverview;