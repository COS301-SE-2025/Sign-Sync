import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import PreferenceManager from "../components/PreferenceManager";

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
      <div className={`flex h-screen ${isDarkMode ? "text-white" : "text-black"}`} style={{ background: isDarkMode
                                                                                                    ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
                                                                                                    : 'linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)'}}>
        <div className="w-64 flex-shrink-0">
            <SideNavbar />
        </div>
      
        <div className="flex-1 h-screen overflow-y-auto relative">
            
            {/* Blur when not logged in */}
            <div className={!this.state.user ? "blur-sm" : ""}>
              <main className="flex flex-col flex-1 items-center w-full max-w-4xl mx-auto p-6 sm:p-8 md:p-12 space-y-8">
                {!showCongratulations ? (
                    <>
                    <header className="text-center space-y-2">
                        <h1 className="text-5xl font-extrabold text-white">Practise words</h1>
                        <p className="text-3xl text-white">
                            Please make the sign for: <span className="text-red-600 font-bold">{currentWord.toUpperCase()}</span>
                        </p>
                        </header>

                    <section className="flex justify-center w-full">
                        <div className="w-auto rounded-lg overflow-hidden border border-gray-300 dark:border-gray-700 shadow-lg">
                            <Camera
                              defaultGestureMode={true}
                              gestureModeFixed={true}
                              onPrediction={this.handlePrediction}
                              width={700}
                              height={400}
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

                          {!isLastWord && (
                              <button
                                onClick={this.handleNext}
                                disabled={!success}
                                className={`px-5 py-2 rounded-md transition ${success ? "bg-indigo-600 text-white hover:bg-indigo-700" : "bg-gray-300 text-gray-600 cursor-not-allowed"}`}
                              >
                                Next
                              </button>
                          )}

                          {isLastWord && success && (
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
              </main>
            </div>

            {/* Login required overlay */}
            {!this.state.user && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className={`p-8 rounded-lg shadow-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} z-10 max-w-md text-center`}>
                  <h2 className="text-2xl font-bold mb-4">Login Required</h2>
                  <p className="mb-6">Please log in to practise the alphabet</p>
                  <button
                    onClick={() => window.location.href = '/login'}
                    className={`px-6 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white font-medium transition-colors`}
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
