import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiMail, FiMessageSquare, FiPhone, FiClock, FiAlertCircle } from 'react-icons/fi';

const ContactSupport = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Contact Support</h1>
          
          <div className="bg-blue-50 p-6 rounded-xl border border-blue-100 mb-8">
            <h2 className="text-xl font-semibold text-blue-800 mb-2">We're Here to Help</h2>
            <p className="text-gray-700">Choose the support option that works best for you</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Email Support */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="bg-blue-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiMail className="text-blue-600 text-xl" />
              </div>
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Email Support</h2>
              <p className="text-gray-600 text-sm mb-4">
                Get help within 24 hours for non-urgent issues
              </p>
              <div className="text-blue-600 font-medium">support@signsync.com</div>
            </div>

            {/* Live Chat */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="bg-green-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiMessageSquare className="text-green-600 text-xl" />
              </div>
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Live Chat</h2>
              <p className="text-gray-600 text-sm mb-2">
                Mon-Fri: 9AM-5PM EST
              </p>
              <p className="text-gray-600 text-sm mb-4">
                Instant connection with our team
              </p>
              <button className="bg-green-600 hover:bg-green-700 text-white text-sm font-medium py-1.5 px-4 rounded-lg transition-colors">
                Start Chat
              </button>
            </div>

            {/* Phone Support */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
              <div className="bg-red-100 p-3 rounded-full w-12 h-12 flex items-center justify-center mb-4">
                <FiPhone className="text-red-600 text-xl" />
              </div>
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Phone Support</h2>
              <p className="text-gray-600 text-sm mb-2">
                24/7 for critical issues
              </p>
              <p className="text-gray-600 text-sm mb-4">
                Enterprise customers only
              </p>
              <div className="text-red-600 font-medium">+1 (800) 555-0199</div>
            </div>
          </div>

          <div className="bg-yellow-50 p-6 rounded-xl border border-yellow-100">
            <h2 className="text-xl font-semibold text-gray-800 mb-3 flex items-center">
              <FiAlertCircle className="mr-2 text-yellow-600" />
              Before Contacting Support
            </h2>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Check our <a href="/troubleshooting" className="text-blue-600 hover:underline">Troubleshooting Guide</a></li>
              <li>Have your account email ready</li>
              <li>Note any error messages you're receiving</li>
              <li>Prepare screenshots if applicable</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSupport;