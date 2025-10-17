import React from "react";
import SideNavbar from "../components/sideNavbar";
import Camera from "../components/Camera";
import LearnAvatar from "../components/LearnAvatar";
import PreferenceManager from "../components/PreferenceManager";
import AchievementsManager from "../components/AchievementsManager";
import AchievementChecker from "../components/AchievementChecker";

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

        // this.setState({ user });

        this.setState({ user });
        // make sure achievements are loaded for this user
        try { await AchievementsManager.initialize(); } catch {}
    }

    //*******************************
    awardLetterAchievements = async (completedCount) => {
        const { user, alphabet } = this.state;
        if (!user) return;
        try {
            // Always ensure manager is ready
            if (!AchievementsManager.userID) {
                await AchievementsManager.initialize();
            }

            const current = AchievementsManager.getAchievements() || [];
            const toAdd = [];

            // First Letter (ID 2)
            if (completedCount >= 1 && !current.includes(2)) toAdd.push(2);
            // All Letters (ID 5)
            if (completedCount >= alphabet.length && !current.includes(5)) toAdd.push(5);

            if (toAdd.length) {
                await AchievementsManager.addAchievements(toAdd);
                // also compute Bronze/Silver/Gold/Platinum based on base set
                await AchievementChecker.checkAchievements(user.userID);
            }
        } catch (e) {
            console.error("Failed to award letter achievements:", e);
        }
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
            }, () => {
                this.awardLetterAchievements(this.state.completedLetters.size);
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
        const { alphabet, currentIndex, success, showCongratulations, user } = this.state;
        const currentLetter = alphabet[currentIndex];
        const isLastLetter = currentIndex === alphabet.length - 1;

        // return (
        //     <div className={`flex min-h-screen ${isDarkMode ? "text-white" : "text-black"}`} style={{ background: isDarkMode
        //             ? "linear-gradient(135deg, #080C1A, #172034)"
        //             : '#f5f5f5'}}>
            
        //     <div className="w-64 flex-shrink-0 h-screen">
        //             <SideNavbar />
        //         </div>

        //         <div className="flex-1 relative h-screen overflow-y-auto justify-center flex items-center">
                    
        //             {/* Blur when not logged in */}
        //             <div className={!this.state.user ? "blur-sm" : ""}>
        //                 <div className="w-full max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8">
        //                     <div
        //                         className={`w-full p-4 sm:p-6 rounded-xl shadow-md dark:shadow-lg transition-all duration-300`}
        //                         style={{
        //                             backgroundColor: isDarkMode ? "#1B2432" : "#f5f5f5",
        //                             border: isDarkMode ? "1px solid #2A3445" : "1px solid #D8CFC2",
        //                         }}
        //                     >
        //                     {!showCongratulations ? (
        //                         <>
        //                             <header className="text-center space-y-2 py-5">
        //                                 <h1 className="text-5xl font-extrabold">Learn the Alphabet</h1>
        //                                 <p className="text-3xl">
        //                                     Current letter:{" "}
        //                                     <span className="text-yellow-400 font-bold">{currentLetter.toUpperCase()}</span>
        //                                 </p>
        //                             </header>

        //                             <section className="flex-row flex justify-center items-start space-x-6">
                                        
        //                                 {/* avatar side */}
        //                                 <div className="flex-none border-r border-gray-400 pr-10">
        //                                     <LearnAvatar key={currentLetter} sentence={currentLetter} compact />
        //                                 </div>

        //                                 {/* camera side */}
        //                                 <div className="flex-none w-[500px] flex flex-col items-center space-y-4">
        //                                     <Camera
        //                                         defaultGestureMode={false}
        //                                         gestureModeFixed={true}
        //                                         onPrediction={this.handlePrediction}
        //                                         width={500}
        //                                         height={300}
        //                                     />
        //                                     <div className="flex flex-row items-center space-x-3">
        //                                         <button
        //                                             onClick={this.handlePrev}
        //                                             disabled={currentIndex === 0}
        //                                             className={`px-5 py-2 rounded-md transition ${currentIndex === 0 ? "bg-gray-300 text-gray-600 cursor-not-allowed" : "bg-indigo-600 text-white hover:bg-indigo-700"}`}
        //                                         >
        //                                             Previous
        //                                         </button>

        //                                         <button
        //                                             onClick={isLastLetter ? this.handleFinish : this.handleNext}
        //                                             disabled={!success}
        //                                             className={`px-5 py-2 rounded-md transition ${success ? "bg-indigo-600 text-white hover:bg-indigo-700" : "bg-gray-300 text-gray-600 cursor-not-allowed"}`}
        //                                         >
        //                                             {isLastLetter ? "Finish" : "Next"}
        //                                         </button>

        //                                         {success && (
        //                                             <p className="text-green-500 font-semibold text-2xl">✔ Well Done!</p>
        //                                         )}
        //                                     </div>
        //                                 </div>

        //                             </section>
        //                         </>
        //                     ) : (
        //                         <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
        //                             <section className="flex flex-col items-center justify-center space-y-6 bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg max-w-lg text-center">
        //                                 <h2 className="text-4xl font-bold text-green-600">
        //                                     🎉 Congratulations! You have signed all the letters correctly! 🎉
        //                                 </h2>

        //                                 <button
        //                                     onClick={this.handleReset}
        //                                     className="px-6 py-3 bg-indigo-600 text-white rounded-lg text-xl font-semibold hover:bg-indigo-700 transition-colors shadow-md"
        //                                 >
        //                                     Start Again
        //                                 </button>
        //                             </section>
        //                         </div>
        //                     )}
        //                     </div>
        //                 </div>
        //             </div>

        //              {/* Login required overlay */}
        //             {!this.state.user && (
        //             <div className="absolute inset-0 flex items-center justify-center">
        //                 <div className={`p-8 rounded-lg shadow-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} z-10 max-w-md text-center`}>
        //                     <h2 className="text-2xl font-bold mb-4">Login Required</h2>
        //                     <p className="mb-6">Please log in to Learn the alphabet</p>
        //                     <button
        //                         onClick={() => window.location.href = '/login'}
        //                         className={`px-6 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white font-medium transition-colors`}
        //                     >
        //                         Go to Login
        //                     </button>
        //                 </div>
        //             </div>
        //             )}
        //         </div>
        //     </div>
        // );

        return (
            <div
                className={`flex min-h-screen ${isDarkMode ? "text-white" : "text-black"}`}
                style={{
                    background: isDarkMode
                        ? "linear-gradient(135deg, #080C1A, #172034)"
                        : "#f5f5f5",
                }}
            >
                {/* Sidebar stays exactly as before */}
                <div className="w-64 flex-shrink-0 h-screen">
                    <SideNavbar />
                </div>

                {/* Right side */}
                <div className="flex-1 relative h-screen overflow-y-auto justify-center flex items-center">
                    {this.state.user ? (
                        //render main content when logged in
                        <div>
                            <div className="w-full max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8">
                                <div
                                    className={`w-full p-4 sm:p-6 rounded-xl shadow-md dark:shadow-lg transition-all duration-300`}
                                    style={{
                                        backgroundColor: isDarkMode ? "#1B2432" : "#f5f5f5",
                                        border: isDarkMode ? "1px solid #2A3445" : "1px solid #D8CFC2",
                                    }}
                                >
                                    {!showCongratulations ? (
                                        <>
                                            <header className="text-center space-y-2 py-5">
                                                <h1 className="text-5xl font-extrabold">Learn the Alphabet</h1>
                                                <p className="text-3xl">
                                                    Current letter:{" "}
                                                    <span className="text-yellow-400 font-bold">
                                                        {currentLetter.toUpperCase()}
                                                    </span>
                                                </p>
                                            </header>

                                            <section className="flex-row flex justify-center items-start space-x-6">
                                                <div className="flex-none border-r border-gray-400 pr-10">
                                                    <LearnAvatar key={currentLetter} sentence={currentLetter} compact />
                                                </div>

                                                <div className="flex-none w-[500px] flex flex-col items-center space-y-4">
                                                    <Camera
                                                        defaultGestureMode={false}
                                                        gestureModeFixed={true}
                                                        onPrediction={this.handlePrediction}
                                                        width={500}
                                                        height={300}
                                                    />
                                                    <div className="flex flex-row items-center space-x-3">
                                                        <button
                                                            onClick={this.handlePrev}
                                                            disabled={currentIndex === 0}
                                                            className={`px-5 py-2 rounded-md transition ${
                                                                currentIndex === 0
                                                                    ? "bg-gray-300 text-gray-600 cursor-not-allowed"
                                                                    : "bg-indigo-600 text-white hover:bg-indigo-700"
                                                            }`}
                                                        >
                                                            Previous
                                                        </button>

                                                        <button
                                                            onClick={isLastLetter ? this.handleFinish : this.handleNext}
                                                            disabled={!success}
                                                            className={`px-5 py-2 rounded-md transition ${
                                                                success
                                                                    ? "bg-indigo-600 text-white hover:bg-indigo-700"
                                                                    : "bg-gray-300 text-gray-600 cursor-not-allowed"
                                                            }`}
                                                        >
                                                            {isLastLetter ? "Finish" : "Next"}
                                                        </button>

                                                        {success && (
                                                            <p className="text-green-500 font-semibold text-2xl">✔ Well Done!</p>
                                                        )}
                                                    </div>
                                                </div>
                                            </section>
                                        </>
                                    ) : (
                                        <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
                                            <section className="flex flex-col items-center justify-center space-y-6 bg-white dark:bg-gray-800 p-8 rounded-xl shadow-lg max-w-lg text-center">
                                                <h2 className="text-4xl font-bold text-green-600">
                                                    🎉 Congratulations! You have signed all the letters correctly! 🎉
                                                </h2>

                                                <button
                                                    onClick={this.handleReset}
                                                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg text-xl font-semibold hover:bg-indigo-700 transition-colors shadow-md"
                                                >
                                                    Start Again
                                                </button>
                                            </section>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ) : (
                        //render only login prompt when NOT logged in
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div
                                className={`p-8 rounded-lg shadow-xl ${isDarkMode ? "bg-gray-800" : "bg-white"} border ${isDarkMode ? "border-gray-700" : "border-gray-200"} z-10 max-w-md text-center`}
                            >
                                <h2 className="text-2xl font-bold mb-4">Login Required</h2>
                                <p className="mb-6">Please log in to Learn the alphabet</p>
                                <button
                                    onClick={() => (window.location.href = "/login")}
                                    className={`px-6 py-2 rounded-lg ${isDarkMode ? "bg-blue-600 hover:bg-blue-700" : "bg-blue-500 hover:bg-blue-600"} text-white font-medium transition-colors`}
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

export default LearnAlphabetPage;
