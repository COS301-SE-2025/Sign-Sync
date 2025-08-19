import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import PreferenceManager from "../components/PreferenceManager";

class PractiseAlphabetPage extends React.Component 
{
  constructor(props) 
  {
    super(props);

    this.initialState = {
      currentIndex: 0,
      //alphabet: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"],
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

  handleSkip = () => 
  {
    this.setState((prevState) => ({
      currentIndex: Math.min(prevState.currentIndex + 1, prevState.alphabet.length - 1),
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
      <div className={`${isDarkMode ? "bg-gray-900 text-white" : "bg-white text-gray-900"} flex h-screen`}>
        <div className="w-64 flex-shrink-0">
            <SideNavbar />
        </div>

        <div className="flex-1 h-screen overflow-y-auto">
            <main className="flex flex-col flex-1 items-center w-full max-w-4xl mx-auto p-6 sm:p-8 md:p-12 space-y-8">
            {!showCongratulations ? (
                <>
                <header className="text-center space-y-2">
                    <h1 className="text-3xl sm:text-4xl font-extrabold">Practise the Alphabet</h1>
                    <p className="text-lg sm:text-xl">
                        Please make the sign for: <span className="text-indigo-600 font-bold">{currentLetter.toUpperCase()}</span>
                    </p>
                    </header>

                <section className="flex justify-center w-full">
                    <div className="w-[90%] sm:w-[70%] md:w-[60%] max-w-md aspect-square rounded-lg overflow-hidden border border-gray-300 dark:border-gray-700 shadow-lg">
                        <Camera
                          defaultGestureMode={false}
                          gestureModeFixed={true}
                          onPrediction={this.handlePrediction}
                          className="w-full h-full object-cover"
                        />
                    </div>
                    </section>

                <section className="flex flex-col items-center space-y-4 w-full">
                    {success && (
                      <p className="text-green-500 font-semibold text-lg">✔ Well Done!</p>
                    )}

                    <nav className="flex gap-4 justify-center">
                      <button
                          onClick={this.handlePrev}
                          disabled={currentIndex === 0}
                          className={`px-5 py-2 rounded-md transition ${currentIndex === 0 ? "bg-gray-300 text-gray-600 cursor-not-allowed" : "bg-indigo-600 text-white hover:bg-indigo-700"}`}
                      >
                          Previous
                      </button>

                      <button
                          onClick={this.handleSkip}
                          className="px-5 py-2 rounded-md bg-yellow-400 text-black hover:bg-yellow-500 transition"
                      >
                          Skip
                      </button>

                      {!isLastLetter && (
                          <button
                            onClick={this.handleNext}
                            disabled={!success}
                            className={`px-5 py-2 rounded-md transition ${success ? "bg-indigo-600 text-white hover:bg-indigo-700" : "bg-gray-300 text-gray-600 cursor-not-allowed"}`}
                          >
                            Next
                          </button>
                      )}

                      {isLastLetter && success && (
                          <button
                            onClick={this.handleFinish}
                            className="px-5 py-2 rounded-md bg-green-600 text-white hover:bg-green-700 transition"
                          >
                            Finish
                          </button>
                      )}
                    </nav>
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
                    Back
                </button>
                </section>
            )}
            </main>
        </div>
      </div>
    );
  }
}

export default PractiseAlphabetPage;
