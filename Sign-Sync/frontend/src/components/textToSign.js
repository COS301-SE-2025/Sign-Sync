import React from "react";

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
            sentence:""
        };
    }


    processSentence = (event) =>{
        var sentence = event.target.value.toUpperCase();
        this.setState({sentence: sentence});
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
            console.log(index + " : " + this.state.sentence.at(index));
            this.setState({signIndex: index});
        }
        const endTime = new Date().getTime();

        console.log("Elapsed time:", endTime - startTime, "milliseconds");
    }

    render()
    {
        return (
            <div className = "text-center mx-auto   ">
                <div >
                    <button onClick={this.cycleLeft}> ⬅️ </button>
                    <img width={300} height={200} src={this.signs[this.state.signIndex]} alt={"Sign Image"}/>
                    <button onClick={this.cycleRight}> ➡️ </button>
                </div>
                <input className="inline outline outline-2 outline-red-500" type={"text"} onChange={this.processSentence}/>
            </div>
        )
    }
}

export default TextToSign;
