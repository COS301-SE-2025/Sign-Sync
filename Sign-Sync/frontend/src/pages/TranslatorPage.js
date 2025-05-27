import React from "react";
import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import SpeechToTextBox from "../components/SpeechToTextBox";

class TranslatorPage extends React.Component
{
    render()
    {
        return (
            // <Camera />
            <div className="flex flex-row h-screen">
                <SideNavbar />

                <div className="flex-1 p-4 overflow-auto">
                    <Camera />

                    {/* textbook with mic button */}
                    <SpeechToTextBox />
                </div>
            </div>
        )
    }
}

export default TranslatorPage;
