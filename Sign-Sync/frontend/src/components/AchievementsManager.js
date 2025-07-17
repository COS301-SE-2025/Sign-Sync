/**
 * Manages user achievements including loading, updating, and tracking progress
 */
class AchievementsManager {
    static achievements = [];
    static userID = null;

    static async initialize() {
        const user = JSON.parse(localStorage.getItem('user'));
        
        if (!user || !user.userID) {
            console.warn("No user logged in - achievements not loaded");
            return false;
        }

        this.userID = user.userID;

        try {
            const response = await fetch(`/userApi/achievements/${this.userID}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Expecting an array of completed achievement IDs (numbers)
            const completedAchievementIds = await response.json();
            
            if (Array.isArray(completedAchievementIds)) {
                this.achievements = completedAchievementIds;
                return true;
            } else {
                console.warn("Invalid achievements data format - expected array");
                this.achievements = [];
                return false;
            }
        } catch (error) {
            console.error("Failed to load achievements:", error);
            this.achievements = [];
            return false;
        }
    }

    static getAchievements() {
        return this.achievements; // Returns array of completed achievement IDs
    }

    /**
     * Updates achievements on the server and locally
     * @param {Array} newAchievements Updated array of achievements
     * @returns {Promise<boolean>} Returns true if update was successful
     */
    static async updateAchievements(newAchievements) {
    if (!this.userID) {
      console.error("No user ID available");
      return false;
    }

    try {
      const response = await fetch(`/userApi/achievements/${this.userID}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ achievements: newAchievements })
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const data = await response.json();
      this.achievements = data.achievements || newAchievements;
      return true;
    } catch (error) {
      console.error("Update failed:", error);
      return false;
    }
  }

    /**
     * Adds a new achievement to the user's achievements
     * @param {Object} achievement The achievement to add
     * @returns {Promise<boolean>} Returns true if addition was successful
     */
    static async addAchievement(achievement) {
        if (!achievement || typeof achievement !== 'object') {
            console.error("Invalid achievement object");
            return false;
        }

        const newAchievements = [...this.achievements, achievement];
        return await this.updateAchievements(newAchievements);
    }
}

export default AchievementsManager;