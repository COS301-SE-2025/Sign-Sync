import React from "react";
import SideNavbar from "../components/sideNavbar";
import { FaQuestionCircle, FaBook, FaVideo, FaEnvelope, FaFileAlt, FaLightbulb } from "react-icons/fa";
import { Link } from "react-router-dom";
import PreferenceManager from "../components/PreferenceManager";

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
            <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
                <div>
                    <SideNavbar />
                </div>

                <div className="flex-1 overflow-y-auto p-8">
                    <div className="max-w-4xl mx-auto">
                        <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Help Menu</h1>

                        {/* Search Help */}
                        <div className="mb-8">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search help articles..."
                                    className={`w-full p-4 pl-12 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 
                                        ${isDarkMode ? "bg-gray-800 border-gray-600 text-white placeholder-gray-400" : "bg-white border-gray-300"}`}
                                    value={searchQuery}
                                    onChange={this.handleSearchChange}
                                />
                                <FaQuestionCircle className={`absolute left-4 top-4 text-xl ${isDarkMode ? "text-gray-400" : "text-gray-400"}`} />
                            </div>
                        </div>

                        {/* Search Results */}
                        {searchQuery.length > 0 && (
                            <div className={`mb-8 rounded-lg shadow-sm border overflow-hidden 
                                ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
                                <h2 className={`p-4 border-b font-semibold ${isDarkMode ? "border-gray-700 text-white" : "border-gray-100 text-gray-800"}`}>
                                    Search Results for "{searchQuery}"
                                </h2>
                                {filteredArticles.length > 0 ? (
                                    filteredArticles.map((article, index) => (
                                        <div 
                                            key={index} 
                                            className={`p-4 ${isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-50"} 
                                                ${index !== filteredArticles.length - 1 ? (isDarkMode ? "border-b border-gray-700" : "border-b border-gray-100") : ""}`}
                                        >
                                            <Link to={article.path} className={`flex justify-between items-center ${isDarkMode ? "text-white hover:text-blue-400" : "text-gray-800 hover:text-blue-600"}`}>
                                                <div>
                                                    <h3 className="font-medium">{article.title}</h3>
                                                    <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>{article.category}</p>
                                                </div>
                                                <svg className={`w-5 h-5 ${isDarkMode ? "text-gray-400" : "text-gray-400"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                                                </svg>
                                            </Link>
                                        </div>
                                    ))
                                ) : (
                                    <div className={`p-4 ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                                        No articles found matching your search.
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Category Cards (Dark mode applied) */}
                        {searchQuery.length === 0 && (
                            <>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                                    {/* Helper for consistent card styling */}
                                    {[
                                        {
                                            icon: <FaBook className="text-blue-600 text-xl" />,
                                            bgIcon: "bg-blue-100",
                                            title: "Getting Started",
                                            desc: "New to our product? Start with these guides.",
                                            links: [
                                                { to: "/productOverview", text: "Product overview" },
                                                { to: "/firstStepsTutorial", text: "First steps tutorial" },
                                                { to: "/settingUpYourAccount", text: "Setting up your account" },
                                            ]
                                        },
                                        {
                                            icon: <FaVideo className="text-purple-600 text-xl" />,
                                            bgIcon: "bg-purple-100",
                                            title: "Tutorials & Videos",
                                            desc: "Watch step-by-step video guides.",
                                            links: [
                                                { to: "/basicFeatures", text: "Basic features walkthrough" },
                                                { to: "/advancedTechniques", text: "Advanced techniques" },                                            ]
                                        },
                                        {
                                            icon: <FaFileAlt className="text-green-600 text-xl" />,
                                            bgIcon: "bg-green-100",
                                            title: "FAQs",
                                            desc: "Find answers to common questions.",
                                            links: [
                                                { to: "/accountSettings", text: "Account settings" },
                                                { to: "/billingQuestions", text: "Billing questions" },
                                                { to: "/troubleshooting", text: "Troubleshooting" },
                                            ]
                                        },
                                        {
                                            icon: <FaEnvelope className="text-yellow-600 text-xl" />,
                                            bgIcon: "bg-yellow-100",
                                            title: "Contact Support",
                                            desc: "Can't find what you need? We're here to help.",
                                            links: [
                                                { to: "/emailSupport", text: "Email support" },
                                                { to: "/liveChatSupport", text: "Live chat" },
                                                { to: "/phoneSupport", text: "Phone support" },
                                            ]
                                        }
                                    ].map((section, i) => (
                                        <div key={i} className={`p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow border 
                                            ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
                                            <div className="flex items-center mb-4">
                                                <div className={`${section.bgIcon} p-3 rounded-full mr-4`}>
                                                    {section.icon}
                                                </div>
                                                <h2 className={`text-xl font-semibold ${isDarkMode ? "text-white" : "text-gray-800"}`}>{section.title}</h2>
                                            </div>
                                            <p className={`${isDarkMode ? "text-gray-400" : "text-gray-600"} mb-4`}>{section.desc}</p>
                                            <ul className="space-y-2">
                                                {section.links.map((link, j) => (
                                                    <li key={j}><Link to={link.to} className="text-blue-600 hover:underline">{link.text}</Link></li>
                                                ))}
                                            </ul>
                                        </div>
                                    ))}
                                </div>

                                {/* Popular Articles */}
                                <div className="mb-8">
                                    <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                                        <FaLightbulb className="text-yellow-500 mr-2" /> Popular Help Articles
                                    </h2>
                                    <div className={`rounded-lg shadow-sm border overflow-hidden 
                                        ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
                                        {[
                                            { title: "How to reset your password", path: "/passwordReset" },
                                            { title: "Understanding your dashboard", path: "/dashboardGuide" },
                                            { title: "Exporting your data", path: "/dataExport" },
                                            { title: "Integrations with other tools", path: "/integrations" },
                                            { title: "Privacy and security settings", path: "/privacySettings" }
                                        ].map((article, index) => (
                                            <div 
                                                key={index} 
                                                className={`p-4 ${isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-50"} 
                                                    ${index !== 0 ? (isDarkMode ? "border-t border-gray-700" : "border-t border-gray-100") : ""}`}
                                            >
                                                <Link to={article.path} className={`flex justify-between items-center ${isDarkMode ? "text-white hover:text-blue-400" : "text-gray-800 hover:text-blue-600"}`}>
                                                    <span>{article.title}</span>
                                                    <svg className={`w-5 h-5 ${isDarkMode ? "text-gray-400" : "text-gray-400"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                        <div className={`p-6 rounded-lg border ${isDarkMode ? "bg-blue-950 border-blue-800 text-white" : "bg-blue-50 border-blue-100 text-gray-800"}`}>
                            <h2 className="text-xl font-semibold mb-2">Still need help?</h2>
                            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} mb-4`}>Our support team is available 24/7 to assist you.</p>
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