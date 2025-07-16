import React from "react";
import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import TextToSign from "../components/textToSign";
import PreferenceManager from "../components/PreferenceManager";

class LearnAlphabetPage extends React.Component 
{
  render() 
  {
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

    return (
      <div className={`flex h-screen ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
        <SideNavbar />
        <div className="flex-1 flex flex-col items-center justify-center overflow-y-auto px-6 py-10">
          <h1 className="text-2xl font-bold mb-4">Learn the Alphabet</h1>
          
          <div>
            <TextToSign />
          </div>
          
          <hr />

          <div className="flex space-x-4">
            <Camera />
          </div>
        </div>
      </div>
    );
  }
}

export default LearnAlphabetPage;