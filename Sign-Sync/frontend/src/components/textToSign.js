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
        this.signs = [
            require("../assets/Alphabet/A.png"),
            require("../assets/Alphabet/B.png"),
            require("../assets/Alphabet/C.png"),
            require("../assets/Alphabet/D.png"),
            require("../assets/Alphabet/E.png"),
            require("../assets/Alphabet/F.png"),
            require("../assets/Alphabet/G.png"),
            require("../assets/Alphabet/H.png"),
            require("../assets/Alphabet/I.png"),
            require("../assets/Alphabet/J.png"),
            require("../assets/Alphabet/K.png"),
            require("../assets/Alphabet/L.png"),
            require("../assets/Alphabet/M.png"),
            require("../assets/Alphabet/N.png"),
            require("../assets/Alphabet/O.png"),
            require("../assets/Alphabet/P.png"),
            require("../assets/Alphabet/Q.png"),
            require("../assets/Alphabet/R.png"),
            require("../assets/Alphabet/S.png"),
            require("../assets/Alphabet/T.png"),
            require("../assets/Alphabet/U.png"),
            require("../assets/Alphabet/V.png"),
            require("../assets/Alphabet/W.png"),
            require("../assets/Alphabet/X.png"),
            require("../assets/Alphabet/Y.png"),
            require("../assets/Alphabet/Z.png")
        ];
        this.state = {
            index:0,
            signIndex:0,
            sentence:"",
            textToBeSent:"",
            mic : false
        };
        this.letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
    }

    processSentence = (event) =>{
        let sentence = event.target.value.toLowerCase();
        this.setState({sentence: sentence});
    }

    sendText = () => {
        let sentence = this.state.sentence;
        if(sentence !== ""){
            this.setState({textToBeSent: sentence});
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
            inputType = <SpeechToTextBox />;
        }else{
            inputType = <input className="text-center w-3/4 text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2.5 justify-center flex flex-grow min-h-[60px] " type={"text"} onChange={this.processSentence} />;
        }

        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        return (

            <div className={`${isDarkMode ? "bg-gray-900 text-white" : "bg-white text-black"}`}>
                <div className={`${isDarkMode ? "bg-gray-800" : "bg-white"} p-2 rounded-lg mb-2 items-center mx-auto`}>
                    <img className="w-[300px] h-[400px] object-contain mx-auto" src={this.state.signIndex === -1 ? null : this.signs[this.state.signIndex]} alt={"No sign available"}/>
                </div>
                <div className={`flex items-center border rounded-lg px-4 py-2 ${isDarkMode ? "bg-gray-800 text-white" : "bg-gray-200 text-black"}`}>
                    <button className="w-1/3 bg-[#801E20] text-[#FFFFFD] p-2 min-h-[60px]" onClick={this.cycleLeft}> Previous Character</button>
                    <h1 className="w-1/3 text-center text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2 justify-center flex flex-grow min-h-[60px]">{this.state.signIndex === -1 ? " " : this.letters[this.state.signIndex]}</h1>
                    <button className="w-1/3 bg-[#801E20] text-[#FFFFFD] p-2 min-h-[60px]" onClick={this.cycleRight}> Next Character </button>
                </div>
                <div className={`flex items-center border rounded-lg px-4 py-2 ${isDarkMode ? "bg-gray-800 text-white" : "bg-gray-200 text-black"}`}>
                    <button className="bg-gray-300 p-3.5 border-2 border-black"><img src={conversationIcon} className="w-8 h-8" alt={"Conversation"}/></button>
                    {inputType}
                    <button onClick={this.changeMic} className="bg-gray-300 p-3.5 border-2 border-black"><img src={mic? MicOn : MicOff} className="w-8 h-8" alt={"Speaker"}/></button>
                    {inputType}
                    <button onClick={this.sendText} className="bg-gray-300 p-3.5 border-2 border-black"><img src={Submit} className="w-8 h-8" alt={"Submit"}/></button>
                </div>
            </div>
        )
    }
}

export default TextToSign;
