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
                <SideNavbar />

                <main className="flex-1 w-full flex">
                    <div className="max-w-6xl mx-auto px-6 py-10 w-full my-auto flex flex-col gap-6">
                        <header className="mb-6">
                            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                                {translatorType === "SignText" ? "Translator" : "Avatar"}
                            </h1>
                        </header>

                        {/* {translatorType === "SignText" ? <TranslatorCamera /> : <TextToSign />} */}
                        {translatorMode}

                        {/* BUTTONS AT BOTTOM */}
                        <div className="flex justify-center gap-4">
                            <button
                                onClick={this.changeType}
                                className={`px-6 py-3 font-bold rounded-xl transition
            ${translatorType === "SignText"
                                        ? "bg-blue-600 hover:bg-blue-700 text-white"
                                        : "bg-indigo-600 hover:bg-indigo-700 text-white"}`}
                            >
                                Translator
                            </button>
                            <button
                                onClick={this.changeType}
                                className={`px-6 py-3 font-bold rounded-xl transition
            ${translatorType === "SignText"
                                        ? "bg-indigo-600 hover:bg-indigo-700 text-white"
                                        : "bg-blue-600 hover:bg-blue-700 text-white"}`}
                            >
                                Avatar
                            </button>
                        </div>
                    </div>
                </main>
            </div>
        )
    }
}

export default TranslatorPage;
