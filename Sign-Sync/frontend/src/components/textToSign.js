import React from "react";
import Submit from "../assets/SubmitArrow.png";
import MicOn from "../assets/MicOn.png";
import MicOff from "../assets/MuteOff.png";
import SpeechToTextBox from "../components/SpeechToTextBox";
import AvatarViewport from "../components/AvatarViewport";

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
        let sentence = this.state.sentence;
        console.log(sentence);
        const aslGlossFunction = async () => {
            try {
                const request = await fetch("http://localhost:8007/api/asl/translate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({sentence: sentence})
                });

                const response = await request.json();
                this.setState({textToBeSent: response.gloss});
                this.setState({timestamp: Date.now()});
            } catch (err) {
                console.error("Failed to fetch prediction:", err);
            }
        }
        if(sentence !== ""){
            aslGlossFunction();
        }
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
                                    background: isDarkMode ? '#36454f' : '#e5e7eb',
                                    color: isDarkMode ? 'white' : 'black',
                                }}
                                type={"text"} value={this.state.sentence} onChange={this.processSentence} />;
        }

        //decide avatar size based on compact mode
        const avatarWidth = this.props.compact ? 500 : 700;
        const avatarHeight = this.props.compact ? 400 : 500;
        
        return (
            <div className= 'p-2 rounded-lg' style={{ background: isDarkMode ? '#36454f' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}>
                <div className="flex flex-col items-center w-full" style={{ maxWidth: avatarWidth}}>
                    {/* Avatar */}
                    <div className= 'w-full p-2 rounded-lg mb-2' style={{ background: isDarkMode ? '#36454f' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}>
                        <AvatarViewport
                            input={this.state.textToBeSent}
                            trigger={this.state.timestamp}
                            width={avatarWidth}
                            height={avatarHeight}
                        />
                    </div>

                    {/* Controls */}
                    <div className= 'flex items-center w-full border rounded-lg px-4 py-2' style={{ background: isDarkMode ? '#36454f' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}>
                        <button onClick={this.changeMic} className="p-3.5 border-2 border-black" style={{ background: isDarkMode ? '#36454f' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}>
                            <img src={mic ? MicOn : MicOff} className="w-8 h-8" alt="Mic"/>
                        </button>
                        {inputType}
                        <button onClick={this.sendText} className="p-3.5 border-2 border-black" style={{ background: isDarkMode ? '#36454f' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}>
                            <img src={Submit} className="w-8 h-8" alt="Submit"/>
                        </button>
                    </div>
                    
                    
                </div>
            </div>
        )
    }
}

export default TextToSign;
