import React from 'react';
import { FiCreditCard, FiDollarSign, FiRefreshCw, FiHelpCircle } from 'react-icons/fi';

const BillingQuestions = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Billing & Subscription</h1>
          
          <div className="space-y-6">
            {/* Payment Methods */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiCreditCard className="mr-2 text-blue-600" />
                Payment Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Accepted Methods</h3>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Visa, Mastercard, American Express</li>
                    <li>• PayPal</li>
                    <li>• Bank transfers (Enterprise only)</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-2">Security</h3>
                  <p className="text-gray-600 text-sm">
                    All payments are processed through PCI-compliant systems. We never store your full card details.
                  </p>
                </div>
              </div>
            </div>

            {/* Subscription Plans */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiDollarSign className="mr-2 text-purple-600" />
                Subscription Management
              </h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-800 mb-1">Changing Plans</h3>
                  <p className="text-gray-600 text-sm">
                    Upgrade or downgrade at any time. Prorated credits will be applied.
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-gray-800 mb-1">Cancellation Policy</h3>
                  <p className="text-gray-600 text-sm">
                    Cancel anytime with no penalties. Your access continues until the end of the billing period.
                  </p>
                </div>
              </div>
            </div>

            {/* Common Issues */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiHelpCircle className="mr-2 text-red-600" />
                Troubleshooting
              </h2>
              <div className="space-y-3">
                <div className="flex items-start">
                  <FiRefreshCw className="text-yellow-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium text-gray-800">Failed Payments</h3>
                    <p className="text-gray-600 text-sm">
                      Update your payment method or contact your bank. We'll automatically retry for 3 days.
                    </p>
                  </div>
                </div>
                <div className="flex items-start">
                  <FiHelpCircle className="text-blue-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium text-gray-800">Invoice Requests</h3>
                    <p className="text-gray-600 text-sm">
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