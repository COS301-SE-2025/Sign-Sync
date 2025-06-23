import React from 'react';
import { FiPhone, FiClock, FiGlobe } from 'react-icons/fi';

const PhoneSupport = () => {
  return (
    <section className="flex h-screen overflow-hidden bg-gray-50">
      <div>
        <SideNavbar />
      </div>

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Phone Support</h1>
          
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-8">
            <div className="flex items-start mb-6">
              <div className="bg-red-100 p-3 rounded-full mr-4">
                <FiPhone className="text-red-600 text-xl" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-800">Direct Line Assistance</h2>
                <p className="text-gray-600">For urgent issues requiring immediate resolution</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Contact Numbers</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-800 font-medium mb-1">United States</p>
                  <p className="text-gray-600 mb-4">+1 (800) 555-0199</p>
                  
                  <p className="text-gray-800 font-medium mb-1">International</p>
                  <p className="text-gray-600">+1 (617) 555-0182</p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Hours of Operation</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-600 mb-2 flex items-center">
                    <FiClock className="mr-2" />
                    <span className="font-medium">Weekdays:</span> 8:00 AM - 8:00 PM EST
                  </p>
                  <p className="text-gray-600 flex items-center">
                    <FiClock className="mr-2" />
                    <span className="font-medium">Weekends:</span> 10:00 AM - 4:00 PM EST
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-100">
              <h3 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                <FiGlobe className="mr-2 text-yellow-600" />
                Important Notes
              </h3>
              <ul className="list-disc pl-5 space-y-1 text-gray-600">
                <li>Phone support is for Enterprise customers only</li>
                <li>Have your customer ID ready (found in Account Settings)</li>
                <li>Average wait time: 7 minutes during peak hours</li>
                <li>For quicker service, use our callback option</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PhoneSupport;