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
                const request = await fetch("http://localhost:8002/translate", {
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
        var mic = this.state.mic;
        let inputType;
        if(mic){
            inputType = <SpeechToTextBox onSpeechInput={this.processSpeech}/>;
        }else{
             inputType = <input className="text-center w-3/4 text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2.5 justify-center flex flex-grow min-h-[60px] " type={"text"} value={this.state.sentence} onChange={this.processSentence} />;
        }

        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        //decide avatar size based on compact mode
        const avatarHeight = this.props.compact ? "40vh" : "65vh";
        const avatarWidth = this.props.compact ? "60vh" : "130vh";
        
        return (

            <div className={`${isDarkMode ? "bg-gray-900 text-white" : "bg-white text-black"}`}>
                <div className={`${isDarkMode ? "bg-gray-800" : "bg-white"} p-2 rounded-lg mb-2 items-center mx-auto`}>
                    <AvatarViewport input={this.state.textToBeSent} trigger={this.state.timestamp} height={avatarHeight} width={avatarWidth}/>
                </div>
                <div className={`flex items-center border rounded-lg px-4 py-2 ${isDarkMode ? "bg-gray-800 text-white" : "bg-gray-200 text-black"}`}>
                    <button onClick={this.changeMic} className="bg-gray-300 p-3.5 border-2 border-black"><img src={mic? MicOn : MicOff} className="w-8 h-8" alt={"Speaker"}/></button>
                    {inputType}
                    <button onClick={this.sendText} className="bg-gray-300 p-3.5 border-2 border-black"><img src={Submit} className="w-8 h-8" alt={"Submit"}/></button>
                </div>
            </div>
        )
    }
}

export default TextToSign;
