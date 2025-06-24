import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiLock, FiMail, FiClock } from 'react-icons/fi';

const PasswordReset = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Password Reset</h1>
          
          <div className="bg-red-50 p-6 rounded-xl border border-red-100 mb-8">
            <h2 className="text-xl font-semibold text-red-800 mb-2">Secure Account Recovery</h2>
            <p className="text-gray-700">Regain access to your account if you've forgotten your password</p>
          </div>

          <div className="space-y-6">
            {/* Reset Process */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiLock className="mr-2 text-blue-500" />
                Reset Instructions
              </h2>
              <ol className="list-decimal pl-5 space-y-3 text-gray-600">
                <li>
                  <span className="font-medium">Go to the login page</span> and click "Forgot password"
                </li>
                <li>
                  <span className="font-medium">Enter your email address</span> associated with your account
                </li>
                <li>
                  <span className="font-medium">Check your email</span> for a password reset link (check spam folder)
                </li>
                <li>
                  <span className="font-medium">Click the link</span> which will open our secure password reset page
                </li>
                <li>
                  <span className="font-medium">Create a new password</span> following our security requirements
                </li>
                <li>
                  <span className="font-medium">Sign in</span> with your new credentials
                </li>
              </ol>
            </div>

            {/* Security Information */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                <FiMail className="mr-2 text-green-500" />
                Security Notes
              </h2>
              <div className="space-y-3 text-gray-600">
                <div className="flex items-start">
                  <FiClock className="text-yellow-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium text-gray-800">Link Expiration</h3>
                    <p className="text-sm">
                      Password reset links expire after 1 hour for your security
                    </p>
                  </div>
                </div>
                <div className="flex items-start">
                  <FiLock className="text-red-500 mr-3 mt-0.5 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium text-gray-800">Password Requirements</h3>
                    <ul className="list-disc pl-5 text-sm space-y-1">
                      <li>Minimum 8 characters</li>
                      <li>At least 1 number</li>
                      <li>At least 1 special character</li>
                      <li>Cannot be a previously used password</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Troubleshooting */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Troubleshooting</h2>
              <div className="space-y-3 text-gray-600">
                <p>
                  <span className="font-medium">No email received?</span> Wait 5 minutes and check spam folder before requesting another.
                </p>
                <p>
                  <span className="font-medium">Link not working?</span> Copy the entire URL from the email into your browser.
                </p>
                <p>
                  <span className="font-medium">Still having trouble?</span> Contact our support team for assistance.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PasswordReset;