import React from "react";
// import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import TextToSign from "../components/textToSign";
import Swap from "../assets/Swap-icon.png"
import PreferenceManager from "../components/PreferenceManager";
import TranslatorCamera from "../components/TranslatorCamera";

class TranslatorPage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            type: "SignText"
        }
    }

    changeType = () => {
        { this.state.type === "SignText" ? this.setState({ type: "TextSign" }) : this.setState({ type: "SignText" }) }
    }

    render() {
        let translatorMode;
        let translatorType = this.state.type;
        if (translatorType === "SignText") {
            translatorMode = <div>
                <TranslatorCamera />
            </div>
        } else {
            translatorMode = <div>
                <TextToSign />
            </div>
        }

        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        return (
            <div
                className={`min-h-screen flex ${isDarkMode ? "text-white" : "text-black"}`}
                style={{
                    background: isDarkMode
                        ? "linear-gradient(135deg, #0A0F1E, #172034)"
                        : "#f5f5f8",
                }}
            >
                
            </div>
        )
    }
}

export default TranslatorPage;
