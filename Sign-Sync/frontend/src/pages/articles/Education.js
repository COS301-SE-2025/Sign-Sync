import React from "react";
import SideNavbar from "../../components/sideNavbar";
import { FaTrophy, FaBook, FaRunning } from "react-icons/fa";

const Education = () => {
        return (
            <section className="flex h-screen overflow-hidden bg-gray-50">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Main Content */}
                <div className="flex-1 overflow-y-auto p-8">
                    <div className="max-w-4xl mx-auto">
                        <h1 className="text-3xl font-bold text-gray-800 mb-6">Education Tab Guide</h1>
                        
                        {/* Achievement Section */}
                        <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100 mb-8">
                            <div className="flex items-center mb-4">
                                <div className="bg-yellow-100 p-3 rounded-full mr-4">
                                    <FaTrophy className="text-yellow-600 text-xl" />
                                </div>
                                <h2 className="text-xl font-semibold text-gray-800">Achievements</h2>
                            </div>
                            <p className="text-gray-600 mb-4">Track your learning progress and earned badges.</p>
                            <ul className="space-y-2">
                                <li><span className="font-medium">Progress Tracking:</span> See completion percentages for each category</li>
                                <li><span className="font-medium">Badges:</span> Earn rewards for milestones</li>
                                <li><span className="font-medium">Streaks:</span> Maintain daily learning streaks</li>
                            </ul>
                        </div>

                        {/* Learn Section */}
                        <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100 mb-8">
                            <div className="flex items-center mb-4">
                                <div className="bg-blue-100 p-3 rounded-full mr-4">
                                    <FaBook className="text-blue-600 text-xl" />
                                </div>
                                <h2 className="text-xl font-semibold text-gray-800">Learn</h2>
                            </div>
                            
                            <div className="mb-4">
                                <h3 className="font-medium text-gray-800 mb-2">Alphabet</h3>
                                <p className="text-gray-600 mb-2">Master sign language letters with interactive lessons.</p>
                                <ul className="list-disc pl-5 text-gray-600">
                                    <li>Video demonstrations for each letter</li>
                                    <li>Slow-motion replays</li>
                                    <li>Practice exercises</li>
                                </ul>
                            </div>
                            
                            <div>
                                <h3 className="font-medium text-gray-800 mb-2">Words</h3>
                                <p className="text-gray-600 mb-2">Learn vocabulary organized by categories.</p>
                                <ul className="list-disc pl-5 text-gray-600">
                                    <li>Common phrases and greetings</li>
                                    <li>Themed vocabulary (food, family, etc.)</li>
                                    <li>Record and compare feature</li>
                                </ul>
                            </div>
                        </div>

                        {/* Practice Section */}
                        <div className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border border-gray-100 mb-8">
                            <div className="flex items-center mb-4">
                                <div className="bg-green-100 p-3 rounded-full mr-4">
                                    <FaRunning className="text-green-600 text-xl" />
                                </div>
                                <h2 className="text-xl font-semibold text-gray-800">Practice</h2>
                            </div>
                            <p className="text-gray-600 mb-4">Test your knowledge with interactive exercises.</p>
                            
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <h3 className="font-medium text-gray-800 mb-2">Flashcards</h3>
                                    <p className="text-gray-600 text-sm">Sign the displayed word or phrase</p>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <h3 className="font-medium text-gray-800 mb-2">Multiple Choice</h3>
                                    <p className="text-gray-600 text-sm">Match signs to their meanings</p>
                                </div>
                                <div className="bg-gray-50 p-4 rounded-lg">
                                    <h3 className="font-medium text-gray-800 mb-2">Freeform</h3>
                                    <p className="text-gray-600 text-sm">Receive feedback on your signing</p>
                                </div>
                            </div>
                        </div>

                        {/* Help Section */}
                        <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                            <h2 className="text-xl font-semibold text-gray-800 mb-2">Need Help With Education Features?</h2>
                            <p className="text-gray-600 mb-4">Our support team can answer any questions about learning tools.</p>
                            <button className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                                Contact Support
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        );
}

export default Education;