import React from 'react';
import SideNavbar from '../../components/sideNavbar';
import { FiCamera, FiMic, FiUser, FiAward } from 'react-icons/fi';
import PreferenceManager from "../../components/PreferenceManager";

const BasicFeatures = () => {
  const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

  // return (
  //   <section className="flex h-screen overflow-hidden bg-gray-50">
  //     <SideNavbar />
  //     <div className="flex-1 overflow-y-auto p-8">
  //       <div className="max-w-4xl mx-auto">
  //         <h1 className="text-3xl font-bold text-gray-800 mb-6">Basic Features Walkthrough</h1>
          
  //         <div className="bg-green-50 p-6 rounded-xl border border-green-100 mb-8">
  //           <h2 className="text-xl font-semibold text-green-800 mb-2">Getting Started with SignSync</h2>
  //           <p className="text-gray-700">Learn how to use the core features of our platform</p>
  //         </div>

  //         <div className="space-y-6">
  //           {/* Translation Section */}
  //           <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
  //             <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
  //               <FiCamera className="mr-2 text-blue-500" />
  //               Sign-to-Speech Translation
  //             </h2>
  //             <ol className="list-decimal pl-5 space-y-3 text-gray-600">
  //               <li>Click the <span className="font-medium">"Sign"</span> button</li>
  //               <li>Allow camera access when prompted</li>
  //               <li>Position your hands within the frame</li>
  //               <li>Perform signs naturally - the system will translate to text/speech</li>
  //             </ol>
  //             <div className="mt-4 bg-gray-50 p-4 rounded-lg">
  //               <h3 className="font-medium text-gray-800 mb-2">Tips for Best Results</h3>
  //               <ul className="list-disc pl-5 space-y-1 text-sm text-gray-600">
  //                 <li>Ensure good lighting on your hands</li>
  //                 <li>Keep background simple and uncluttered</li>
  //                 <li>Start with basic signs before trying complex sentences</li>
  //               </ul>
  //             </div>
  //           </div>

  //           {/* Speech Section */}
  //           <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
  //             <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
  //               <FiMic className="mr-2 text-purple-500" />
  //               Speech-to-Sign Translation
  //             </h2>
  //             <ol className="list-decimal pl-5 space-y-3 text-gray-600">
  //               <li>Click the <span className="font-medium">"Speech"</span> button</li>
  //               <li>Allow microphone access when prompted</li>
  //               <li>Speak clearly into your microphone</li>
  //               <li>View the avatar's sign language translation</li>
  //             </ol>
  //           </div>

  //           {/* Account Section */}
  //           <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
  //             <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
  //               <FiUser className="mr-2 text-green-500" />
  //               Your Account Dashboard
  //             </h2>
  //             <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  //               <div>
  //                 <h3 className="font-medium text-gray-800 mb-2">Key Features</h3>
  //                 <ul className="text-gray-600 space-y-2 text-sm">
  //                   <li>• View your translation history</li>
  //                   <li>• Access saved phrases</li>
  //                   <li>• Check your subscription status</li>
  //                 </ul>
  //               </div>
  //               <div>
  //                 <h3 className="font-medium text-gray-800 mb-2">Quick Actions</h3>
  //                 <ul className="text-gray-600 space-y-2 text-sm">
  //                   <li>• Edit your profile</li>
  //                   <li>• Change password</li>
  //                   <li>• Manage connected devices</li>
  //                 </ul>
  //               </div>
  //             </div>
  //           </div>
  //         </div>
  //       </div>
  //     </div>
  //   </section>
  // );

  return (
    <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
      <SideNavbar />
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Basic Features Walkthrough
          </h1>

          <div className={`p-6 rounded-xl border mb-8 
            ${isDarkMode ? "bg-green-950 border-green-800" : "bg-green-50 border-green-100"}`}>
            <h2 className="text-xl font-semibold text-green-500 mb-2">Getting Started with SignSync</h2>
            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
              Learn how to use the core features of our platform
            </p>
          </div>

          <div className="space-y-6">
            {/* Sign-to-Speech Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiCamera className="mr-2 text-blue-500" />
                Sign-to-Speech Translation
              </h2>
              <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Click the <span className="font-medium">"Sign"</span> button</li>
                <li>Allow camera access when prompted</li>
                <li>Position your hands within the frame</li>
                <li>Perform signs naturally - the system will translate to text/speech</li>
              </ol>
              <div className={`mt-4 p-4 rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-50"}`}>
                <h3 className={`font-medium mb-2 ${isDarkMode ? "text-white" : "text-gray-800"}`}>Tips for Best Results</h3>
                <ul className={`list-disc pl-5 space-y-1 text-sm ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                  <li>Ensure good lighting on your hands</li>
                  <li>Keep background simple and uncluttered</li>
                  <li>Start with basic signs before trying complex sentences</li>
                </ul>
              </div>
            </div>

            {/* Speech-to-Sign Section */}
            <div className={`p-6 rounded-xl shadow-sm border 
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-100"}`}>
              <h2 className={`text-xl font-semibold mb-4 flex items-center ${isDarkMode ? "text-white" : "text-gray-800"}`}>
                <FiMic className="mr-2 text-purple-500" />
                Speech-to-Sign Translation
              </h2>
              <ol className={`list-decimal pl-5 space-y-3 ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
                <li>Click the <span className="font-medium">"Speech"</span> button</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak clearly into your microphone</li>
                <li>View the avatar's sign language translation</li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default BasicFeatures;