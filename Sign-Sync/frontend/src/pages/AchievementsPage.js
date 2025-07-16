import React from "react";
import SideNavbar from "../components/sideNavbar";
import PreferenceManager from "../components/PreferenceManager";
import FirstAlphabetAchievment from "../assets/FirstAlphabetAchievment(Easy).png";
import FirstLetterAchievment from "../assets/FirstLetterAchievment(Easy).png";
import LearnedTheAlphabetAchievment from "../assets/LearnedTheAlphabetAchievment(Medium).png";
import LearnedTheDictionaryAchievment from "../assets/LearnedTheDictionaryAchievment(Hard).png";
import AchievementsManager from "../components/AchievementsManager";

class AchievementsPage extends React.Component {
  state = {
    achievements: [],
    isLoading: true,
    error: null
  };

  async componentDidMount() {
    try {
      // Initialize the achievements manager
      const initialized = await AchievementsManager.initialize();
      
      if (initialized) {
        // Get achievements from the manager
        const achievements = AchievementsManager.getAchievements();
        this.setState({ 
          achievements: this.mapToLocalFormat(achievements),
          isLoading: false 
        });
      } else {
        this.setState({ 
          isLoading: false,
          error: "Failed to load achievements" 
        });
      }
    } catch (error) {
      console.error("Error loading achievements:", error);
      this.setState({ 
        isLoading: false,
        error: error.message 
      });
    }
  }

  // Helper to map API achievements to local format with images
  mapToLocalFormat(completedAchievementIds = []) {
    const defaultAchievements = [
      {
        id: 1,
        name: "First Letter",
        description: "Successfully perform your first letter sign",
        image: FirstAlphabetAchievment,
        completed: completedAchievementIds.includes(1), 
        difficulty: "easy"
      },
      {
        id: 2,
        name: "First Word",
        description: "Successfully perform your first word sign",
        image: FirstLetterAchievment,
        completed: completedAchievementIds.includes(2),
        difficulty: "easy"
      },
      {
        id: 3,
        name: "Learned the Alphabet",
        description: "Successfully perform all letter sign",
        image: LearnedTheAlphabetAchievment,
        completed: completedAchievementIds.includes(3),
        difficulty: "medium"
      },
      {
        id: 4,
        name: "Learned the Dictionary",
        description: "Successfully perform all word sign",
        image: LearnedTheDictionaryAchievment,
        completed: completedAchievementIds.includes(4),
        difficulty: "hard"
      }
    ];

    return defaultAchievements;
  }

  render() {
    const { achievements, isLoading, error } = this.state;
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
      hard: "bg-red-100 text-red-800"
    };

    const darkModeDifficultyColors = {
      easy: "bg-green-900 text-green-200",
      medium: "bg-yellow-900 text-yellow-200",
      hard: "bg-red-900 text-red-200"
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
              <p className="text-red-500 mb-4">Error loading achievements: {error}</p>
              <button 
                onClick={() => window.location.reload()}
                className={`px-4 py-2 rounded ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white`}
              >
                Retry
              </button>
            </div>
          </div>
        </section>
      );
    }

    return (
      <section className={`flex h-screen overflow-hidden ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-50 text-black"}`}>
        <div>
          <SideNavbar />
        </div>
        
        <div className="flex-1 p-6 overflow-y-auto">
          <h1 className="text-3xl font-bold mb-6">Achievements</h1>
          
          {/* Achievements Grid */}
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
          
          {/* Progress Tracker */}
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