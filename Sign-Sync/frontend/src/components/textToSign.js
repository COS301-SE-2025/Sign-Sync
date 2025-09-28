import React from "react";
import Submit from "../assets/SubmitArrow.png";
import MicOn from "../assets/MicOn.png";
import MicOff from "../assets/MuteOff.png";
import hourGlass from "../assets/HourGlass.png";
import SpeechToTextBox from "../components/SpeechToTextBox";
import AvatarViewport from "../components/AvatarViewport";
import { toast } from "react-toastify";

import PreferenceManager from "./PreferenceManager";

class TextToSign extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            timestamp: Date.now(),
            sentence: "",
            textToBeSent: "",
            emotionToBeSent: "",
            mic: false,
            isTranslating: false,
            animationPlaying : false
        };
    }

    componentDidMount() {
        if (this.props.sentence) {
            this.setState({ sentence: this.props.sentence }, () => {
                this.sendText();
            })
        }
    }

    processSentence = (event) => {
        let sentence = event.target.value.toLowerCase();
        this.setState({ sentence: sentence });
    }

    processSpeech = (text) => {
        let sentence = text.toLowerCase();
        this.setState({ sentence: sentence });
    }

    sendText = () => {
        let sentence = this.state.sentence;
        console.log(sentence);

        const aslGlossFunction = async () => {
            try {
                const request = await fetch("http://localhost:8007/api/asl/translate", {
                    //const request = await fetch("https://apigateway-evbsd4dmhbbyhwch.southafricanorth-01.azurewebsites.net/api/asl/translate", { //deployment version
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ sentence: sentence })
                });

                if (!request.ok) {                                    
                    let serverMsg = "";                             
                    try {                                           
                        const e = await request.json();                   
                        serverMsg = e?.message || "";                 
                    } catch { }                                      
                    if (request.status === 429) {                       
                        const ra = request.headers.get("Retry-After");    
                        const secs = ra && /^\d+$/.test(ra) ? parseInt(ra, 10) : 45; 
                        console.error("Rate limited.");               
                        // Don’t push undefined into the avatar props: 
                        this.setState({ textToBeSent: "", emotionToBeSent: "Neutral" }); 
                        toast?.error?.(serverMsg || `Too many requests. Try again in ${secs}s.`); // optional 
                    } else {                                        
                        toast?.error?.(serverMsg || "Translation failed."); 
                        this.setState({ textToBeSent: "", emotionToBeSent: "Neutral" }); 
                    }                                               
                    return;                                         
                }

                // const response = await request.json();
                // this.setState({ textToBeSent: response.gloss });
                // this.setState({ emotionToBeSent: response.emotion });
                // this.setState({ timestamp: Date.now() });

                const res = await request.json();                       
                this.setState({                                     
                    textToBeSent: res?.gloss || "",                
                    emotionToBeSent: res?.emotion || "Neutral",    
                    timestamp: Date.now(),
                });

            }
            catch (err) {
                console.error("Failed to fetch prediction:", err);

                this.setState({                                   
                    textToBeSent: "",                               
                    emotionToBeSent: "Neutral",                     
                });                                               
                toast?.error?.("Network error during translation."); 
            }
            finally {                                         
                this.setState({ isTranslating: false });          
            }
        }
        if (sentence !== "") {
            aslGlossFunction();
        }
    }

    changeMic = () => {
        this.setState({ mic: !this.state.mic });
    }

    render() {
        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";

        var mic = this.state.mic;
        let inputType;
        if (mic) {
            inputType = <SpeechToTextBox onSpeechInput={this.processSpeech} />;
        } else {
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
                            <AvatarViewport
                                input={this.state.textToBeSent}
                                emotion={this.state.emotionToBeSent}
                                trigger={this.state.timestamp}
                                width={avatarWidth}
                                height={avatarHeight}
                                playing={(value) => {
                                    this.setState({ animationPlaying: value });
                                }}
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
                            disabled={this.state.animationPlaying}
                            onClick={this.sendText}
                            className="p-3.5 border-2 border-black"
                            style={{ background: isDarkMode ? '#353535ff' : '#e5e7eb', color: isDarkMode ? 'white' : 'black' }}
                        >
                            <img src={this.state.animationPlaying? hourGlass: Submit} className="w-8 h-8" alt="Submit" />
                        </button>
                    </div>
                </div>
            </div>
        )
    }
}

export default TextToSign;
