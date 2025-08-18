import React from "react";
import SideNavbar from "../components/sideNavbar";
import { FaQuestionCircle, FaBook, FaVideo, FaEnvelope, FaFileAlt, FaLightbulb } from "react-icons/fa";
import { Link } from "react-router-dom";

class HelpMenuPage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            searchQuery: "",
            filteredArticles: []
        };
    }

    // All help articles data
    articles = [
        {
            title: "Product overview",
            path: "/productOverview",
            category: "Getting Started",
            content: "Learn about our product features"
        },
        {
            title: "First steps tutorial",
            path: "/firstStepsTutorial",
            category: "Getting Started",
            content: "Beginner's guide to using the application"
        },
        {
            title: "Setting up your account",
            path: "/settingUpYourAccount",
            category: "Getting Started",
            content: "How to create and configure your account"
        },
        {
            title: "Basic features walkthrough",
            path: "/basicFeatures",
            category: "Tutorials & Videos",
            content: "Introduction to main features"
        },
        {
            title: "Advanced techniques",
            path: "/advancedTechniques",
            category: "Tutorials & Videos",
            content: "Pro tips for power users"
        },
        {
            title: "Account settings",
            path: "/accountSettings",
            category: "FAQs",
            content: "How to manage your account preferences"
        },
        {
            title: "Billing questions",
            path: "/billingQuestions",
            category: "FAQs",
            content: "Payment and subscription information"
        },
        {
            title: "Troubleshooting",
            path: "/troubleshooting",
            category: "FAQs",
            content: "Solutions to common problems"
        },
        {
            title: "How to reset your password",
            path: "/passwordReset",
            category: "Popular",
            content: "Step-by-step password recovery"
        },
        {
            title: "Understanding your dashboard",
            path: "/dashboardGuide",
            category: "Popular",
            content: "Navigation and feature overview"
        }
    ];

    handleSearchChange = (e) => {
        const query = e.target.value.toLowerCase();
        this.setState({ searchQuery: query });

        if (query.length > 0) {
            const results = this.articles.filter(article => 
                article.title.toLowerCase().includes(query) ||
                article.content.toLowerCase().includes(query) ||
                article.category.toLowerCase().includes(query)
            );
            this.setState({ filteredArticles: results });
        } else {
            this.setState({ filteredArticles: [] });
        }
    };

    render() {
        const { searchQuery, filteredArticles } = this.state;
        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        return (
            // <section className="flex h-screen overflow-hidden bg-gray-50">
            <section
                className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
                style={{
                    background: isDarkMode
                        ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
                        : "linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)",
                }}
            >
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
                                    value={searchQuery}
                                    onChange={this.handleSearchChange}
                                />
                                <FaQuestionCircle className="absolute left-4 top-4 text-gray-400 text-xl" />
                            </div>
                        </div>

                        {/* Search Results */}
                        {searchQuery.length > 0 && (
                            <div className="mb-8 bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden">
                                <h2 className="p-4 border-b border-gray-100 font-semibold text-gray-800">
                                    Search Results for "{searchQuery}"
                                </h2>
                                {filteredArticles.length > 0 ? (
                                    filteredArticles.map((article, index) => (
                                        <div 
                                            key={index} 
                                            className={`p-4 hover:bg-gray-50 ${index !== filteredArticles.length - 1 ? 'border-b border-gray-100' : ''}`}
                                        >
                                            <Link to={article.path} className="flex justify-between items-center text-gray-800 hover:text-blue-600">
                                                <div>
                                                    <h3 className="font-medium">{article.title}</h3>
                                                    <p className="text-sm text-gray-500">{article.category}</p>
                                                </div>
                                                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                                                </svg>
                                            </Link>
                                        </div>
                                    ))
                                ) : (
                                    <div className="p-4 text-gray-500">
                                        No articles found matching your search.
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Help Categories (only shown when not searching) */}
                        {searchQuery.length === 0 && (
                            <>
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
                                            <li><Link to="/productOverview" className="text-blue-600 hover:underline">Product overview</Link></li>
                                            <li><Link to="/firstStepsTutorial" className="text-blue-600 hover:underline">First steps tutorial</Link></li>
                                            <li><Link to="/settingUpYourAccount" className="text-blue-600 hover:underline">Setting up your account</Link></li>
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
                                            <li><Link to="/basicFeatures" className="text-blue-600 hover:underline">Basic features walkthrough</Link></li>
                                            <li><Link to="/advancedTechniques" className="text-blue-600 hover:underline">Advanced techniques</Link></li>
                                            <li><Link to="/productivityTips" className="text-blue-600 hover:underline">Productivity tips</Link></li>
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
                                            <li><Link to="/accountSettings" className="text-blue-600 hover:underline">Account settings</Link></li>
                                            <li><Link to="/billingQuestions" className="text-blue-600 hover:underline">Billing questions</Link></li>
                                            <li><Link to="/troubleshooting" className="text-blue-600 hover:underline">Troubleshooting</Link></li>
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
                                            <li><Link to="/emailSupport" className="text-blue-600 hover:underline">Email support</Link></li>
                                            <li><Link to="/liveChatSupport" className="text-blue-600 hover:underline">Live chat</Link></li>
                                            <li><Link to="/phoneSupport" className="text-blue-600 hover:underline">Phone support</Link></li>
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
                                            { title: "How to reset your password", path: "/passwordReset" },
                                            { title: "Understanding your dashboard", path: "/dashboardGuide" },
                                            { title: "Exporting your data", path: "/dataExport" },
                                            { title: "Integrations with other tools", path: "/integrations" },
                                            { title: "Privacy and security settings", path: "/privacySettings" }
                                        ].map((article, index) => (
                                            <div 
                                                key={index} 
                                                className={`p-4 hover:bg-gray-50 ${index !== 0 ? 'border-t border-gray-100' : ''}`}
                                            >
                                                <Link to={article.path} className="flex justify-between items-center text-gray-800 hover:text-blue-600">
                                                    <span>{article.title}</span>
                                                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                                                    </svg>
                                                </Link>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </>
                        )}

                        {/* Still Need Help? */}
                        <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                            <h2 className="text-xl font-semibold text-gray-800 mb-2">Still need help?</h2>
                            <p className="text-gray-600 mb-4">Our support team is available 24/7 to assist you.</p>
                            <Link 
                                to="/contactSupport" 
                                className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
                            >
                                Contact Support
                            </Link>
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}

export default HelpMenuPage;