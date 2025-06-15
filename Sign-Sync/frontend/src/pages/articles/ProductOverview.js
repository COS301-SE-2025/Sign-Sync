import React from "react";

const ProductOverview = () => {
    return (
        <section className="flex h-screen overflow-hidden bg-gray-50">
            {/* Sidebar */}
            <div>
                <SideNavbar />
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto p-8">
                <div className="max-w-4xl mx-auto">
                    <h1 className="text-3xl font-bold text-gray-800 mb-6">Product Overview</h1>
                    
                    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Welcome to [Your Product Name]!</h2>
                        <p className="text-gray-600 mb-4">
                            Our platform is designed to [briefly state the main purpose of your product].
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
                        <h2 className="text-xl font-semibold text-gray-800 mb-4">Key Features</h2>
                        <ul className="list-disc pl-6 space-y-2 text-gray-600">
                            <li><strong>Feature 1:</strong> Description and benefits.</li>
                            <li><strong>Feature 2:</strong> How it improves workflow.</li>
                            <li><strong>Feature 3:</strong> Integration with other tools.</li>
                        </ul>
                    </div>

                    {/* Add more sections as needed */}

                    <div className="bg-blue-50 p-6 rounded-lg border border-blue-100">
                        <h2 className="text-xl font-semibold text-gray-800 mb-2">Next Steps</h2>
                        <p className="text-gray-600 mb-4">Ready to dive in? Check out our tutorials.</p>
                        <button className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                            Getting Started Guide
                        </button>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default ProductOverview;