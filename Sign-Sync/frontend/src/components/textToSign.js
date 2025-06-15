import React from "react";
import conversationIcon from "../assets/conversation.png";
import MicOn from "../assets/MicOn.png";
import MicOff from "../assets/MuteOff.png";
import SpeechToTextBox from "../components/SpeechToTextBox";
import avatarViewport from "../components/AvatarViewport"
import AvatarViewport from "../components/AvatarViewport";

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
            mic : false
        };
        this.letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"];
    }


    processSentence = (event) =>{
        var sentence = event.target.value.toLowerCase();
        if(event.target.value.charAt(-1)!== " "){
            this.setState({sentence: sentence});
        }
        if(sentence.length === 1){
            this.showSign();
        }
    }

    cycleRight = () =>{
        var index = this.state.index+1;
        if(index < this.state.sentence.length){
            this.setState({index:index});
        }
        this.showSign();
    }

    cycleLeft = () =>{
        var index = this.state.index-1;
        if(index >= 0){
            this.setState({index:index});
        }
        this.showSign();
    }

    showSign = () => {
        const startTime = new Date().getTime();
        let sentence = this.state.sentence;
        if(sentence !== ""){
            let index = sentence.charCodeAt(this.state.index) - 65;
            if(index < 26 && index >= 0){
                this.setState({signIndex: index});
            }else{
                this.setState({signIndex: -1});
            }
            this.setState({sentence: sentence});
        }
        const endTime = new Date().getTime();

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
            inputType = <input className="text-center w-3/4 text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2 justify-center flex flex-grow min-h-[60px] " type={"text"} onChange={this.processSentence}/>;
        }
        return (
            <div>
                <div className="bg-white p-2 rounded-lg mb-2 items-center mx-auto">
                    <AvatarViewport input={this.state.sentence}/>
                </div>
                <div className="flex items-center border bg-gray-200 rounded-lg px-4 py-2 ">
                    <button className="w-1/3 bg-[#801E20] text-[#FFFFFD] p-2 min-h-[60px]" onClick={this.cycleLeft}> Previous Character</button>
                    <h1 className="w-1/3 text-center text-4xl font-bold border-2 border-black bg-gray-300 py-2.5 my-2 justify-center flex flex-grow min-h-[60px]">{this.state.signIndex === -1 ? " " : this.letters[this.state.signIndex]}</h1>
                    <button className="w-1/3 bg-[#801E20] text-[#FFFFFD] p-2 min-h-[60px]" onClick={this.cycleRight}> Next Character </button>
                </div>
                <div className="flex items-center border bg-gray-200 rounded-lg px-4 py-2 ">
                    <button className="bg-gray-300 p-3.5 border-2 border-black"><img src={conversationIcon} className="w-8 h-8" alt={"Conversation"}/></button>
                    {inputType}
                    <button onClick={this.changeMic} className="bg-gray-300 p-3.5 border-2 border-black"><img src={mic? MicOn : MicOff} className="w-8 h-8" alt={"Speaker"}/></button>
                </div>
            </div>
        )
    }
}

export default TextToSign;
