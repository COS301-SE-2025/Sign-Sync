import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import PreferenceManager from "../components/PreferenceManager";
import AvatarViewport from "../components/AvatarViewport";

class PractiseAlphabetPage extends React.Component 
{
    constructor(props) 
    {
        super(props);

        this.state = {
            currentIndex: 0,
            alphabet: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
            success: false
        };
    }

    handleNext = () => 
    {
        this.setState(prevState => ({
            currentIndex: Math.min(prevState.currentIndex + 1, prevState.alphabet.length - 1),
            success: false
        }));
    };

    handlePrev = () => 
    {
        this.setState(prevState => ({
            currentIndex: Math.max(prevState.currentIndex - 1, 0),
            success: false
        }));
    };

    handlePrediction = (prediction) => 
    {
        const currentLetter = this.state.alphabet[this.state.currentIndex];

        if(prediction.toLowerCase() === currentLetter) 
        {
            this.setState({ success: true });
        }
    };

    render() 
    {
        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
        const { alphabet, currentIndex, success } = this.state;
        const currentLetter = alphabet[currentIndex];

        return (
            <div className={`flex h-screen ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
                <SideNavbar />
                <div className="flex-1 flex flex-col items-center justify-center overflow-y-auto px-6 py-10 gap-6">
                    <h1 className="text-2xl font-bold">Learn the Alphabet</h1>

                    <AvatarViewport input={currentLetter} trigger={Date.now()} />

                    <div className="text-4xl font-semibold">
                        Please make the sign for: <span className="text-blue-500">{currentLetter.toUpperCase()}</span>
                    </div>

                    <Camera 
                        defaultGestureMode={false} 
                        gestureModeFixed={true}
                        onPrediction={this.handlePrediction} 
                    />

                    <div className="text-xl mt-4">
                        {success ? <span className="text-green-500 font-bold">✔ Well Done!</span> : ""}
                    </div>

                    <div className="flex space-x-4 mt-8">
                        <button 
                            onClick={this.handlePrev} 
                            className="px-6 py-2 bg-gray-300 rounded hover:bg-gray-400"
                        >
                            Previous
                        </button>
                        <button 
                            onClick={this.handleNext} 
                            disabled={!success}
                            className={`px-6 py-2 rounded ${success ? "bg-blue-500 text-white hover:bg-blue-600" : "bg-gray-300 text-gray-500 cursor-not-allowed"}`}
                        >
                            Next
                        </button>
                    </div>
                </div>
            </div>
        );
    }
}

export default PractiseAlphabetPage;
