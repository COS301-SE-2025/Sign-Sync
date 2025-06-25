import React from 'react';
import SideNavbar from "../../components/sideNavbar";
import { FiUser, FiMail, FiLock, FiCheckCircle, FiLogIn } from 'react-icons/fi';

const SettingUpYourAccount = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <div>
        <SideNavbar />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Account Setup</h1>
          
          {/* Registration Section */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiUser className="mr-2 text-blue-600" />
              Create Your Account
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Step-by-Step Guide */}
              <div>
                <ol className="space-y-6">
                  <li className="flex items-start">
                    <div className="flex-shrink-0 bg-blue-100 text-blue-600 rounded-full p-2 mr-4">
                      <FiUser className="text-lg" />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-gray-800 mb-1">Registration Form</h3>
                      <p className="text-gray-600">
                        Fill in the required fields:
                      </p>
                      <ul className="mt-2 space-y-2 text-gray-600">
                        <li className="flex items-start">
                          <FiCheckCircle className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                          <span><span className="font-medium">Username:</span> 6-20 characters (letters, numbers, _)</span>
                        </li>
                        <li className="flex items-start">
                          <FiCheckCircle className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                          <span><span className="font-medium">Email:</span> Valid email address</span>
                        </li>
                        <li className="flex items-start">
                          <FiCheckCircle className="text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                          <span><span className="font-medium">Password:</span> 8+ characters with 1 number</span>
                        </li>
                      </ul>
                    </div>
                  </li>
                  <li className="flex items-start">
                    <div className="flex-shrink-0 bg-blue-100 text-blue-600 rounded-full p-2 mr-4">
                      <FiMail className="text-lg" />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-gray-800 mb-1">Email Verification</h3>
                      <p className="text-gray-600">
                        After submitting, check your email (including spam folder) for a verification link.
                        Click the link to activate your account.
                      </p>
                    </div>
                  </li>
                </ol>
              </div>

              {/* Visual Demo */}
              <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
                <div className="mb-4 text-center">
                  <h3 className="text-lg font-medium text-gray-800 mb-2">Example Registration</h3>
                  <div className="inline-block bg-white p-4 rounded-lg shadow-xs border border-gray-300">
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                        <div className="h-9 bg-gray-100 rounded-md">Help</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                        <div className="h-9 bg-gray-100 rounded-md">help@gmail.com</div>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                        <div className="h-9 bg-gray-100 rounded-md">*****</div>
                      </div>
                      <button className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md transition-colors">
                        Register
                      </button>
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500 text-center">Your information is secured with encryption</p>
              </div>
            </div>
          </div>

          {/* Login Section */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
              <FiLogIn className="mr-2 text-purple-600" />
              Existing Users
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Sign In Process</h3>
                <ol className="space-y-3 text-gray-600">
                  <li className="flex items-start">
                    <span className="bg-purple-100 text-purple-600 rounded-full w-5 h-5 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0 text-sm">1</span>
                    <span>Click <span className="font-semibold">"Sign in"</span> in the top navigation</span>
                  </li>
                  <li className="flex items-start">
                    <span className="bg-purple-100 text-purple-600 rounded-full w-5 h-5 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0 text-sm">2</span>
                    <span>Enter your registered email and password</span>
                  </li>
                  <li className="flex items-start">
                    <span className="bg-purple-100 text-purple-600 rounded-full w-5 h-5 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0 text-sm">3</span>
                    <span>Click <span className="font-semibold">"Login"</span> to access your dashboard</span>
                  </li>
                </ol>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-medium text-gray-800 mb-3">Troubleshooting</h3>
                <ul className="space-y-2 text-gray-600">
                  <li className="flex items-start">
                    <FiLock className="text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span><span className="font-medium">Forgot password?</span> Use the recovery option</span>
                  </li>
                  <li className="flex items-start">
                    <FiMail className="text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span><span className="font-medium">No verification email?</span> Check spam or resend</span>
                  </li>
                  <li className="flex items-start">
                    <FiUser className="text-blue-500 mr-2 mt-0.5 flex-shrink-0" />
                    <span><span className="font-medium">Account locked?</span> Contact support</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SettingUpYourAccount;