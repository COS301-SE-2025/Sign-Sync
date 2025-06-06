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

                        </div>
                </div>
            </section>
        );
    }
}

export default HelpMenuPage;