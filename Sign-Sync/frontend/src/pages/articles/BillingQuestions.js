import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiCreditCard, FiDollarSign, FiRefreshCw, FiHelpCircle } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const BillingQuestions = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  return (
    // <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
    <section
      className={`flex h-screen overflow-hidden ${isDarkMode ? "text-white" : "text-black"}`}
      style={{
        background: isDarkMode
          ? "linear-gradient(135deg, #080C1A, #172034)"
          : "#f5f5f5",
      }}
    >
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Billing & Subscription
          </h1>

          <div className="space-y-6">
            {/* Payment Methods */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiCreditCard className="mr-2 text-blue-600" />
                Payment Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Accepted Methods</h3>
                  <ul className={`text-sm space-y-1 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    <li>• Visa, Mastercard, American Express</li>
                    <li>• PayPal</li>
                    <li>• Bank transfers (Enterprise only)</li>
                  </ul>
                </div>
                <div>
                  <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Security</h3>
                  <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    All payments are processed through PCI-compliant systems. We never store your full card details.
                  </p>
                </div>
              </div>
            </div>

            {/* Subscription Plans */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiDollarSign className="mr-2 text-purple-600" />
                Subscription Management
              </h2>
              <div className="space-y-4">
                <div>
                  <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Changing Plans</h3>
                  <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    Upgrade or downgrade at any time. Prorated credits will be applied.
                  </p>
                </div>
                <div>
                  <h3 className={`font-medium mb-1 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Cancellation Policy</h3>
                  <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                    Cancel anytime with no penalties. Your access continues until the end of the billing period.
                  </p>
                </div>
              </div>
            </div>

            {/* Common Issues */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiHelpCircle className="mr-2 text-red-600" />
                Troubleshooting
              </h2>
              <div className="space-y-3">
                <div className="flex items-start">
                  <FiRefreshCw className="text-yellow-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className={`font-medium ${isDarkMode ? "text-white" : "text-gray-800"}`}>Failed Payments</h3>
                    <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                      Update your payment method or contact your bank. We'll automatically retry for 3 days.
                    </p>
                  </div>
                </div>
                <div className="flex items-start">
                  <FiHelpCircle className="text-blue-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className={`font-medium ${isDarkMode ? "text-white" : "text-gray-800"}`}>Invoice Requests</h3>
                    <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                      Access past invoices in your account dashboard or contact billing@signsync.com.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default BillingQuestions;