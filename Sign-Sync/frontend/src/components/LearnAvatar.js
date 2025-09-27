import React from "react";
import Submit from "../assets/SubmitArrow.png";
import MicOn from "../assets/MicOn.png";
import MicOff from "../assets/MuteOff.png";
import SpeechToTextBox from "../components/SpeechToTextBox";
import LearnAvatarViewport from "../components/LearnAvatarViewPort";

import PreferenceManager from "./PreferenceManager";

class TextToSign extends React.Component
{
    constructor(props)
    {
        super(props);
        this.state = {
            timestamp: Date.now(),
            sentence:"",
            textToBeSent:"",
            emotionToBeSent:"",
            mic : false
        };
    }

    componentDidMount() {
        if(this.props.sentence) {
            this.setState({sentence: this.props.sentence}, () => {
                this.sendText();
            })
        }
    }

    processSentence = (event) =>{
        let sentence = event.target.value.toLowerCase();
        this.setState({sentence: sentence});
    }

    processSpeech = (text) =>{
        let sentence = text.toLowerCase();
        this.setState({sentence: sentence});
    }

    sendText = () => {
        this.setState({textToBeSent: this.state.sentence});
        this.setState({emotionToBeSent: "Neutral"});
        this.setState({timestamp: Date.now()});
    }

    changeMic = () => {
        this.setState({mic: !this.state.mic});
    }

    render()
    {
        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        var mic = this.state.mic;
        let inputType;
        if(mic){
            inputType = <SpeechToTextBox onSpeechInput={this.processSpeech}/>;
        }else{
             inputType = <input className="w-full text-center text-4xl font-bold border-2 border-black py-2.5 mx-2 min-h-[60px]"
                                style={{
                                    background: isDarkMode ? '#353535ff' : '#e5e7eb',
                                    color: isDarkMode ? 'white' : 'black',
                                }}
                                type={"text"} value={this.state.sentence} onChange={this.processSentence} />;
        }

        //decide avatar size based on compact mode
        const avatarWidth = this.props.compact ? 500 : 700;
        const avatarHeight = this.props.compact ? 400 : 500;
        
        return (
            <div
                className="rounded-2xl p-6 mx-auto"
                style={{
                    background: isDarkMode ? '#1B2432' : '#e5e7eb',
                    color: isDarkMode ? 'white' : 'black',
                    border: isDarkMode ? "1px solid #2A3445" : "1px solid #D8CFC2",
                    // Make the whole card only as wide as the avatar + padding
                    maxWidth: `${avatarWidth + 80}px` // padding/border allowance
                }}
            >
                {/* CHANGED: added mx-auto to center this column */}
                <div className="flex flex-col items-center w-full">

                    {/* Avatar stage */}
                    {/* CHANGED: grid + place-items-center to center the inner box */}
                    <div
                        className="w-full p-2 rounded-lg mb-2 grid place-items-center"
                        style={{ background: isDarkMode ? '#1B2432' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}
                    >
                        {/* CHANGED: explicit box that defines the viewport size */}
                        <div style={{ width: avatarWidth, height: avatarHeight }}>
                            <LearnAvatarViewport
                                input={this.state.textToBeSent}
                                emotion={this.state.emotionToBeSent}
                                trigger={this.state.timestamp}
                                width={avatarWidth}
                                height={avatarHeight}
                            />
                        </div>
                    </div>

                    {/* Controls (already sized to the same column width) */}
                    <div
                        className="flex items-center w-full border rounded-lg px-4 py-2"
                        style={{ background: isDarkMode ? '#353535ff' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}
                    >
                        <button
                            onClick={this.changeMic}
                            className="p-3.5 border-2 border-black"
                            style={{ background: isDarkMode ? '#353535ff' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}
                        >
                            <img src={mic ? MicOn : MicOff} className="w-8 h-8" alt="Mic" />
                        </button>

                        {inputType}

                        <button
                            onClick={this.sendText}
                            className="p-3.5 border-2 border-black"
                            style={{ background: isDarkMode ? '#353535ff' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}
                        >
                            <img src={Submit} className="w-8 h-8" alt="Submit" />
                        </button>
                    </div>
                </div>
            </div>
        )
    }
}

export default TextToSign;
