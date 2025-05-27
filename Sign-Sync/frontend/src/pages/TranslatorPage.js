import React from "react";
import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import TextToSign from "../components/textToSign";
import SpeechToTextBox from "../components/speechToTextBox";

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
                <Camera/>
            </div>
        }else{
            translatorMode = <div>
                <SpeechToTextBox/>
                <TextToSign/>
            </div>
        }
        return (
            <div className={"flex h-screen"}>
                <SideNavbar/>
                <div className={"flex-1 flex items-center justify-center bg-gray-100"}>
                    {translatorMode}
                    <button onClick={this.changeType}>Switch mode</button>
                </div>
            </div>
        )
    }
}

export default TranslatorPage;
