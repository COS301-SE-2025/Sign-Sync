import React from "react";
import SideNavbar from "../components/sideNavbar";
import PreferenceManager from "../components/PreferenceManager";
import FirstAlphabetAchievment from "../assets/FirstAlphabetAchievment(Easy).png";
import FirstLetterAchievment from "../assets/FirstLetterAchievment(Easy).png";
import WelcomeAchievment from "../assets/WelcomeAchievment(Easy).png";
import BronzeAchievment from "../assets/BronzeAchievment(Easy).png";
import LearnedTheAlphabetAchievment from "../assets/LearnedTheAlphabetAchievment(Medium).png";
import SilverAchievment from "../assets/SilverAchievment(Medium).png";
import LearnedTheDictionaryAchievment from "../assets/LearnedTheDictionaryAchievment(Hard).png";
import GoldAchievment from "../assets/GoldAchievment(Hard).png";
import PlatinumAchievment from "../assets/PlatinumAchievment(Ultimate).png";
import AchievementsManager from "../components/AchievementsManager";
import AchievementChecker from "../components/AchievementChecker.js"
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

class AchievementsPage extends React.Component {
  state = {
    achievements: [],
    user: JSON.parse(localStorage.getItem('user')),
    error: null,
    isLoading: true,
    unlockedAchievements: [],
    showUnlockedModal: false
  };

  mapToLocalFormat(completedAchievementIds = []) {
    const defaultAchievements = [
      {
        id: 1,
        name: "Welcome",
        description: "Login for the first time",
        image: WelcomeAchievment,
        completed: completedAchievementIds.includes(1), 
        difficulty: "easy"
      },
      {
        id: 2,
        name: "First Letter",
        description: "Successfully perform your first letter sign",
        image: FirstAlphabetAchievment,
        completed: completedAchievementIds.includes(2), 
        difficulty: "easy"
      },
      {
        id: 3,
        name: "First Word",
        description: "Successfully perform your first word sign",
        image: FirstLetterAchievment,
        completed: completedAchievementIds.includes(3),
        difficulty: "easy"
      },
      {
        id: 4,
        name: "Bronze",
        description: "Have 25% Completion",
        image: BronzeAchievment,
        completed: completedAchievementIds.includes(4),
        difficulty: "easy"
      },
      {
        id: 5,
        name: "Learned the Alphabet",
        description: "Successfully perform all letter sign",
        image: LearnedTheAlphabetAchievment,
        completed: completedAchievementIds.includes(5),
        difficulty: "medium"
      },
      {
        id: 6,
        name: "Silver",
        description: "Have 50% Completion",
        image: SilverAchievment,
        completed: completedAchievementIds.includes(6),
        difficulty: "medium"
      },
      {
        id: 7,
        name: "Learned the Dictionary",
        description: "Successfully perform all word sign",
        image: LearnedTheDictionaryAchievment,
        completed: completedAchievementIds.includes(7),
        difficulty: "hard"
      },
      {
        id: 8,
        name: "Gold",
        description: "Have 75% Completion",
        image: GoldAchievment,
        completed: completedAchievementIds.includes(8),
        difficulty: "hard"
      },
      {
        id: 9,
        name: "Platinum",
        description: "Have 100% Completion",
        image: PlatinumAchievment,
        completed: completedAchievementIds.includes(9),
        difficulty: "ultimate"
      }
    ];

    return defaultAchievements;
  }

  async componentDidMount() {
    try {
      if (!this.state.user) {
        this.setState({ isLoading: false });
        return;
      }

      // Initial load
      await this.loadAchievements();

      // Add event listener for window focus
      window.addEventListener('focus', this.handleWindowFocus);

    } catch (error) {
      console.error("Achievement load error:", error);
      this.setState({ 
        isLoading: false,
        error: error.message 
      });
    }
  }

  componentWillUnmount() {
    // Clean up the event listener when component unmounts
    window.removeEventListener('focus', this.handleWindowFocus);
  }

  handleWindowFocus = async () => {
    try {
      await this.loadAchievements();
    } catch (error) {
      console.error("Error checking achievements on focus:", error);
    }
  };

  loadAchievements = async () => {
    try {
      // 1. Load current achievements
      const initialized = await AchievementsManager.initialize();
      if (!initialized) {
        console.error("AchievementsManager initialization failed");
        throw new Error("Failed to initialize achievements");
      }

      // 2. Check for new achievements
      const totalAchievements = this.mapToLocalFormat([]).length;
      let newAchievements = [];
      
      try {
        newAchievements = await AchievementChecker.checkAchievements(
          this.state.user.userID, 
          totalAchievements
        );
      } catch (checkError) {
        console.error("Error checking achievements:", checkError);
        // Continue even if checking fails - we'll just show existing achievements
      }

      // 3. Refresh if new achievements were added
      if (newAchievements.length > 0) {
        try {
          await AchievementsManager.initialize();
        } catch (reinitError) {
          console.error("Error reinitializing achievements:", reinitError);
        }
      }

      // 4. Get current achievements (even if checking failed)
      const currentAchievements = AchievementsManager.getAchievements();
      
      // 5. Update state
      this.setState({
        achievements: this.mapToLocalFormat(currentAchievements || []),
        isLoading: false,
        unlockedAchievements: newAchievements
      });

      // 6. Show notifications if we got new achievements
      if (newAchievements.length > 0) {
        this.showAchievementNotification(newAchievements);
      }

    } catch (error) {
      console.error("Error in loadAchievements:", error);
      this.setState({ 
        isLoading: false,
        error: error.message 
      });
    }
  };
  
  showAchievementNotification(newAchievementIds) {
    const { achievements } = this.state;
    const unlocked = achievements.filter(a => 
      newAchievementIds.includes(a.id)
    );

    // Show modal with all unlocked achievements
    this.setState({ showUnlockedModal: true });

    // Also show individual toasts
    unlocked.forEach(achievement => {
      toast.success(
        <div>
          <h3 className="font-bold">Achievement Unlocked!</h3>
          <p>{achievement.name}</p>
          <img 
            src={achievement.image} 
            alt={achievement.name}
            className="w-12 h-12 mx-auto mt-2"
          />
        </div>, 
        {
          position: toast.POSITION.TOP_RIGHT,
          autoClose: 5000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true
        }
      );
    });
  }

  renderUnlockedModal() {
    const { unlockedAchievements, achievements } = this.state;
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    const unlocked = achievements.filter(a => unlockedAchievements.includes(a.id));

    return (
      <div className={`fixed inset-0 flex items-center justify-center z-50 ${isDarkMode ? 'bg-black bg-opacity-70' : 'bg-gray-900 bg-opacity-50'}`}>
        <div className={`p-6 rounded-lg shadow-xl w-full max-w-md ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className="text-2xl font-bold mb-4 text-center">New Achievements Unlocked!</h2>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {unlocked.map(achievement => (
              <div key={achievement.id} className="flex items-center p-3 rounded-lg bg-opacity-20 bg-blue-500">
                <img 
                  src={achievement.image} 
                  alt={achievement.name}
                  className="w-16 h-16 mr-4"
                />
                <div>
                  <h3 className="font-semibold">{achievement.name}</h3>
                  <p className="text-sm">{achievement.description}</p>
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => this.setState({ showUnlockedModal: false })}
            className={`mt-6 w-full py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white`}
          >
            Continue
          </button>
        </div>
      </div>
    );
  }

  render() {
    const { achievements, user, error, isLoading, showUnlockedModal } = this.state;
    const isDarkMode = PreferenceManager.getPreferences().displayMode === "Dark Mode";
    
    // Calculate completion percentage
    const completedCount = achievements.filter(a => a.completed).length;
    const completionPercentage = achievements.length > 0 
      ? Math.round((completedCount / achievements.length) * 100)
      : 0;

    // Difficulty color mapping
    const difficultyColors = {
      easy: "bg-green-100 text-green-800",
      medium: "bg-yellow-100 text-yellow-800",
      hard: "bg-red-100 text-red-800",
      ultimate: "bg-purple-100 text-purple-800"
    };

    const darkModeDifficultyColors = {
      easy: "bg-green-900 text-green-200",
      medium: "bg-yellow-900 text-yellow-200",
      hard: "bg-red-900 text-red-200",
      ultimate: "bg-purple-900 text-purple-200"
    };

    if (isLoading) {
      return (
        <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
          <SideNavbar />
          <div className="flex-1 p-6 overflow-y-auto flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        </section>
      );
    }

    if (error) {
      return (
        <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
          <SideNavbar />
          <div className="flex-1 p-6 overflow-y-auto flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-500 mb-4">Error: {error}</p>
              <p className="mb-4">Showing cached achievements (may be outdated)</p>
              <button 
                onClick={this.loadAchievements}
                className={`px-4 py-2 rounded ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white`}
              >
                Retry
              </button>
            </div>
          </div>
        </section>
      );
    }

    if (!user) {
      return (
        <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
          <div>
            <SideNavbar />
          </div>
          
          <div className="flex-1 p-6 overflow-y-auto relative">
            <div className="blur-sm">
              <h1 className="text-3xl font-bold mb-6">Achievements</h1>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                {this.mapToLocalFormat([]).map(achievement => (
                  <div 
                    key={achievement.id}
                    className={`rounded-lg p-4 border transition-all ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}
                  >
                    <div className="flex items-start gap-4">
                      <div className="relative">
                        <img 
                          src={achievement.image} 
                          alt={achievement.name}
                          className="w-16 h-16 object-cover rounded-lg grayscale"
                        />
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-start">
                          <h3 className="font-semibold text-lg">{achievement.name}</h3>
                          <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                            isDarkMode 
                              ? darkModeDifficultyColors[achievement.difficulty] 
                              : difficultyColors[achievement.difficulty]
                          }`}>
                            {achievement.difficulty.charAt(0).toUpperCase() + achievement.difficulty.slice(1)}
                          </span>
                        </div>
                        <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                          {achievement.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className={`mt-8 p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
                <div className="flex justify-between items-center mb-2">
                  <h2 className="font-semibold">Your Progress</h2>
                  <span>0/{achievements.length} (0%)</span>
                </div>
                <div className={`w-full h-4 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-300'}`}>
                  <div 
                    className="h-full rounded-full bg-gradient-to-r from-blue-500 to-green-500"
                    style={{ width: '0%' }}
                  ></div>
                </div>
              </div>
            </div>
            
            <div className="absolute inset-0 flex items-center justify-center">
              <div className={`p-8 rounded-lg shadow-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} z-10 max-w-md text-center`}>
                <h2 className="text-2xl font-bold mb-4">Login Required</h2>
                <p className="mb-6">Please log in to view your achievements</p>
                <button
                  onClick={() => window.location.href = '/login'}
                  className={`px-6 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white font-medium transition-colors`}
                >
                  Go to Login
                </button>
              </div>
            </div>
          </div>
        </section>
      );
    }

    return (
      <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
        {showUnlockedModal && this.renderUnlockedModal()}
        <div>
          <SideNavbar />
        </div>
        
        <div className="flex-1 p-6 overflow-y-auto">
          <h1 className="text-3xl font-bold mb-6">Achievements</h1>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {achievements.map(achievement => (
              <div 
                key={achievement.id}
                className={`rounded-lg p-4 border transition-all ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} ${
                  !achievement.completed ? 'opacity-70' : ''
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className="relative">
                    <img 
                      src={achievement.image} 
                      alt={achievement.name}
                      className={`w-16 h-16 object-cover rounded-lg ${
                        !achievement.completed ? 'grayscale' : ''
                      }`}
                    />
                    {achievement.completed && (
                      <div className="absolute -top-2 -right-2 bg-green-500 rounded-full w-6 h-6 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <h3 className="font-semibold text-lg">{achievement.name}</h3>
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                        isDarkMode 
                          ? darkModeDifficultyColors[achievement.difficulty] 
                          : difficultyColors[achievement.difficulty]
                      }`}>
                        {achievement.difficulty.charAt(0).toUpperCase() + achievement.difficulty.slice(1)}
                      </span>
                    </div>
                    <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      {achievement.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className={`mt-8 p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
            <div className="flex justify-between items-center mb-2">
              <h2 className="font-semibold">Your Progress</h2>
              <span>{completedCount}/{achievements.length} ({completionPercentage}%)</span>
            </div>
            <div className={`w-full h-4 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-300'}`}>
              <div 
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-500"
                style={{ width: `${completionPercentage}%` }}
              ></div>
            </div>
          </div>
        </div>
      </section>
    );
  }
}

export default AchievementsPage;