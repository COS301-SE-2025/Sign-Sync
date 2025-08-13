import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import TextToSign from "../components/textToSign";
import PreferenceManager from "../components/PreferenceManager";

class LearnAlphabetPage extends React.Component 
{
    constructor(props) 
    {
        super(props);

        this.initialState = {
            currentIndex: 0,
            // alphabet: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
            alphabet: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"],
            success: false,
            completedLetters: new Set(),
            showCongratulations: false,
        };

        this.state = { ...this.initialState };
    }

    handleNext = () => 
    {
        this.setState((prevState) => ({
            currentIndex: Math.min(prevState.currentIndex + 1, prevState.alphabet.length - 1),
            success: false,
        }));
    };

    handlePrev = () => 
    {
        this.setState((prevState) => ({
        currentIndex: Math.max(prevState.currentIndex - 1, 0),
        success: false,
        }));
    };

    handleFinish = () => 
    {
        this.setState({ showCongratulations: true });
    };

    handlePrediction = (prediction) => 
    {
        const currentLetter = this.state.alphabet[this.state.currentIndex];

        if(prediction.toLowerCase() === currentLetter) 
        {
            this.setState((prevState) => {
                const newCompleted = new Set(prevState.completedLetters);
                newCompleted.add(currentLetter);

                return {
                    success: true,
                    completedLetters: newCompleted,
                };
            });
        }
    };

    handleReset = () => 
    {
        this.setState({ ...this.initialState });
    };

    render() 
    {
        const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
        const { alphabet, currentIndex, success, showCongratulations } = this.state;
        const currentLetter = alphabet[currentIndex];
        const isLastLetter = currentIndex === alphabet.length - 1;

        return (
            <div className={`flex h-screen ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
                <div className="w-64 flex-shrink-0">
                    <SideNavbar />
                </div>

                <div className="flex-1 h-screen overflow-y-auto">
                    {/* <main className="flex flex-col items-center w-full max-w-4xl mx-auto p-6 sm:p-8 md:p-12 space-y-12"> */}
                    <main className="flex flex-col items-center w-full p-6 sm:p-8 md:p-12 space-y-12">
                        {!showCongratulations ? (
                            <>
                                <header className="text-center space-y-2">
                                    <h1 className="text-5xl font-extrabold">Learn the Alphabet</h1>
                                    <p className="text-3xl">
                                        Current letter:{" "}
                                        <span className="text-indigo-600 font-bold">{currentLetter.toUpperCase()}</span>
                                    </p>
                                </header>

                                <section className="w-full flex justify-center items-center space-x-6">
                                    
                                    {/* avatar side */}
                                    <div className="flex-none border-r border-gray-400 pr-10">
                                        <TextToSign key={currentLetter} sentence={currentLetter} compact />
                                    </div>

                                    {/* camera side */}
                                    <div className="flex-none" style={{ width: '500px' }}>
                                        <Camera
                                            defaultGestureMode={false}
                                            gestureModeFixed={true}
                                            onPrediction={this.handlePrediction}
                                            width={500}
                                            height={450}
                                        />
                                    </div>
                                </section>

                                <section className="flex flex-col items-center space-y-4">
                                    {success && (
                                        <p className="text-green-500 font-semibold text-lg">✔ Well Done!</p>
                                    )}

                                    <button
                                        onClick={this.handlePrev}
                                        disabled={currentIndex === 0}
                                        className={`px-5 py-2 rounded-md transition ${currentIndex === 0 ? "bg-gray-300 text-gray-600 cursor-not-allowed" : "bg-indigo-600 text-white hover:bg-indigo-700"}`}
                                    >
                                        Previous
                                    </button>

                                    <button
                                        onClick={isLastLetter ? this.handleFinish : this.handleNext}
                                        disabled={!success}
                                        className={`px-5 py-2 rounded-md transition ${success ? "bg-indigo-600 text-white hover:bg-indigo-700" : "bg-gray-300 text-gray-600 cursor-not-allowed"}`}
                                    >
                                        {isLastLetter ? "Finish" : "Next"}
                                    </button>
                                </section>
                            </>
                        ) : (
                            <section className="text-center space-y-6">
                                <h2 className="text-4xl font-bold text-green-600">
                                    🎉 Congratulations! You have signed all the letters correctly! 🎉
                                </h2>
                                <button
                                    onClick={this.handleReset}
                                    className="text-indigo-600 underline text-xl hover:text-indigo-800 transition"
                                >
                                    Start Again
                                </button>
                            </section>
                        )}
                    </main>
                </div>
            </div>
        );
    }
}

export default LearnAlphabetPage;
