import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import PreferenceManager from "../components/PreferenceManager";
import EducationTranslatorCamera from "../components/EducationTranslator";

class PractiseWordsPage extends React.Component 
{
  constructor(props) 
  {
    super(props);

    this.initialState = {
      currentIndex: 0,
      words: ["go", "know", "live", "movie", "we", "you", "school", "thank", "tomorrow", "tonight", "watch", "yes", "goodbye", "hello"],
      success: false,
      completedWords: new Set(),
      showCongratulations: false,
      user: null
    };

    this.state = { ...this.initialState };
  }

  async componentDidMount()
  {
    const user = JSON.parse(localStorage.getItem('user'));

    if(!user) 
    {
      this.setState({ user: null });
      return;
    }

    this.setState({ user });
  }

  handleNext = () => 
  {
    this.setState((prevState) => ({
      currentIndex: Math.min(prevState.currentIndex + 1, prevState.words.length - 1),
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
      currentIndex: Math.min(prevState.currentIndex + 1, prevState.words.length - 1),
      success: false,
    }));
  };

  handleFinish = () => 
  {
    this.setState({ showCongratulations: true });
  };

  handlePrediction = (prediction) => 
  {
    const currentWord = this.state.words[this.state.currentIndex];

    if(prediction.toLowerCase() === currentWord) 
    {
      this.setState((prevState) => {
        const newCompleted = new Set(prevState.completedWords);
        newCompleted.add(currentWord);
        
        return {
          success: true,
          completedWords: newCompleted,
        };
      });
    }
  };

  handleReset = () => 
  {
    this.setState((prevState) => ({
        ...this.initialState,
        user: prevState.user,
    }));
  };

  render() 
  {
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    const { words, currentIndex, success, showCongratulations, user} = this.state;
    const currentWord = words[currentIndex];
    const isLastWord = currentIndex === words.length - 1;

    return (
      <div
        className={`flex h-screen ${isDarkMode ? "text-white" : "text-black"}`}
        style={{
          background: isDarkMode
            ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
            : "linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)",
        }}
      >
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0">
          <SideNavbar />
        </div>

        {/* Main content */}
        <div
          className={`flex-1 h-screen overflow-y-auto relative flex items-center justify-center ${
            isDarkMode ? "text-white" : "text-black"
          }`}
        >
          {/* Blur when not logged in */}
          <div className={!this.state.user ? "blur-sm" : ""}>
            <div className="w-full max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8">
              <div
                className={`w-full p-6 sm:p-8 md:p-10 lg:p-12 rounded-xl shadow-md dark:shadow-lg transition-all duration-300`}
                style={{
                  backgroundColor: isDarkMode ? "#1B2432" : "#f5f5f5",
                  border: isDarkMode ? "1px solid #2A3445" : "1px solid #D8CFC2",
                }}
              >
                {!showCongratulations ? (
                  <>
                    {/* Header */}
                    <header className="text-center space-y-2">
                      <h1 className="text-5xl font-extrabold text-white">
                        Practise Words
                      </h1>
                      <p className="text-3xl text-white">
                        Please make the sign for:{" "}
                        <span className="text-yellow-400 font-bold">
                          {currentWord.toUpperCase()}
                        </span>
                      </p>
                    </header>

                    {/* Progress + Camera */}
                    <section className="flex justify-center w-full gap-6 mt-6">
                      {/* Left panel: Progress tracking */}
                      <div
                        className="w-96 rounded-lg border border-gray-300 dark:border-gray-700 shadow-lg p-4 flex flex-col"
                        style={{
                          background: isDarkMode ? "#1e293b" : "#f5f5f5",
                          height: "520px",
                        }}
                      >
                        <div className="space-y-3 overflow-y-auto flex-1">
                          {words.map((word, index) => {
                            const isCurrent = index === currentIndex;
                            const isDone = index < currentIndex;

                            return (
                              <div
                                key={index}
                                className={`flex items-center justify-between px-3 py-1 rounded-md font-medium ${
                                  isCurrent
                                    ? "bg-[#1a436b] text-white"
                                    : isDone
                                    ? "text-green-600"
                                    : "text-gray-800 dark:text-gray-200"
                                }`}
                              >
                                <span className="text-2xl font-bold">{word}</span>

                                {isCurrent && (
                                  <button
                                    onClick={this.handleSkip}
                                    className="ml-2 text-sm bg-yellow-400 text-black px-3 py-1 rounded hover:bg-yellow-500"
                                  >
                                    Skip
                                  </button>
                                )}

                                {isDone && (
                                  <span className="ml-2 text-green-600 font-bold">
                                    ✔
                                  </span>
                                )}
                              </div>
                            );
                          })}
                        </div>

                        {/* Nav buttons */}
                        <div className="flex justify-between mt-4">
                          <button
                            onClick={this.handlePrev}
                            disabled={currentIndex === 0}
                            className={`px-3 py-1 rounded-md transition ${
                              currentIndex === 0
                                ? "bg-gray-300 text-gray-600 cursor-not-allowed"
                                : "bg-indigo-600 text-white hover:bg-indigo-700"
                            }`}
                          >
                            Previous
                          </button>

                          {isLastWord && success ? (
                            <button
                              onClick={this.handleFinish}
                              className="px-4 py-2 rounded-md bg-green-600 text-white hover:bg-green-700 transition"
                            >
                              Finish
                            </button>
                          ) : (
                            <button
                              onClick={this.handleNext}
                              disabled={!success || isLastWord}
                              className={`px-4 py-2 rounded-md transition ${
                                !success || isLastWord
                                  ? "bg-gray-300 text-gray-600 cursor-not-allowed"
                                  : "bg-blue-600 text-white hover:bg-blue-700"
                              }`}
                            >
                              Next
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Right panel: Camera */}
                      <div className="flex-1 rounded-lg overflow-hidden border border-gray-300 dark:border-gray-700 shadow-lg">
                        <EducationTranslatorCamera
                          onPrediction={(prediction) =>
                            this.handlePrediction(prediction)
                          }
                        />
                      </div>
                    </section>

                    {/* Well Done Message */}
                    {success && (
                      <p className="text-green-500 font-semibold text-3xl mt-4 text-center">
                        ✔ Well Done!
                      </p>
                    )}
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
                    <section className="flex flex-col items-center justify-center space-y-6 bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg max-w-lg text-center">
                      <h2 className="text-4xl font-bold text-green-600">
                        🎉 Congratulations! You have signed all the words correctly! 🎉
                      </h2>

                      <button
                        onClick={this.handleReset}
                        className="px-6 py-3 bg-indigo-600 text-white rounded-lg text-xl font-semibold hover:bg-indigo-700 transition-colors shadow-md"
                      >
                        Back
                      </button>
                    </section>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Login overlay */}
          {!this.state.user && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div
                className={`p-8 rounded-lg shadow-xl ${
                  isDarkMode ? "bg-gray-800" : "bg-white"
                } border ${isDarkMode ? "border-gray-700" : "border-gray-200"} z-10 max-w-md text-center`}
              >
                <h2 className="text-2xl font-bold mb-4">Login Required</h2>
                <p className="mb-6">Please log in to practise the words</p>
                <button
                  onClick={() => (window.location.href = "/login")}
                  className={`px-6 py-2 rounded-lg ${
                    isDarkMode
                      ? "bg-blue-600 hover:bg-blue-700"
                      : "bg-blue-500 hover:bg-blue-600"
                  } text-white font-medium transition-colors`}
                >
                  Go to Login
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }
}

export default PractiseWordsPage;
