import React from "react";
// import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import TextToSign from "../components/textToSign";
import Swap from "../assets/Swap-icon.png"
import PreferenceManager from "../components/PreferenceManager";
import TranslatorCamera from "../components/TranslatorCamera";

class TranslatorPage extends React.Component
{
    constructor(props) {
        super(props);
        this.state = {
            type: "SignText"
        }
    }

    changeType = () => {
        {this.state.type === "SignText" ? this.setState({type: "TextSign"}) : this.setState({type: "SignText"})}
    }

    render()
    {
        let translatorMode;
        let translatorType = this.state.type;
        if(translatorType === "SignText"){
            translatorMode = <div>
                <TranslatorCamera/>
            </div>
        }else{
            translatorMode = <div>
                <TextToSign/>
            </div>
        }

        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        return (
            // <div className={`flex h-screen ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <div className={`flex h-screen ${isDarkMode ? "text-white" : "text-black"}`} style={{ background: isDarkMode
                                                                                                    ? "linear-gradient(135deg, #080C1A, #172034)"
                                                                                                    : "#f5f5f5"}}>
                <SideNavbar/>
                <div className={`flex-1 flex flex-col items-center justify-center overflow-y-auto ${isDarkMode ? "text-white" : "text-black"} px-6 py-10`}>
                {/* <div className={`flex-1 flex flex-col items-center justify-center overflow-y-auto ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"} px-6 py-10`}> */}
                    {translatorMode}
                    <div className={"flex mt-2"}>
                        <button onClick={this.changeType} className={`p-2 font-bold w-20 ${translatorType === "SignText" ? "bg-[#801E20] text-[#FFFFFD]": "bg-[#FFFFFD] text-[#801E20]"}  border-gray-600 rounded`}>
                            Sign
                        </button>
                        <button className="px-4" onClick={this.changeType}>
                            <img src={Swap} alt={"Swap"} className="w-8 h-8"/>
                        </button>
                        <button onClick={this.changeType} className={`p-2 font-bold w-20 ${translatorType === "SignText" ? "bg-[#FFFFFD] text-[#801E20]" : "bg-[#801E20] text-[#FFFFFD]"} border-gray-600 rounded`}>
                            Speech
                        </button>
                    </div>
                </div>
            </div>
        )
    }
}

export default TranslatorPage;
