import AchievementsManager from "../components/AchievementsManager";

export default class AchievementChecker {
  static async checkAchievements(userID, totalAchievements) {

    let newAchievements = [];

    try {
      if (!AchievementsManager.userID) {
        await AchievementsManager.initialize();
      }

      // Get current achievements through manager
      const currentAchievements = AchievementsManager.getAchievements();

      // Welcome Achievement (should be handled during login)
      // if (!currentAchievements.includes(1)) {
      //   newAchievements.push(1);
      // }

      // First Alphabet Achievement
      if (!currentAchievements.includes(2)) {
        const hasFirstAlphabet = await this.checkFirstAlphabet(userID);
        if (hasFirstAlphabet) newAchievements.push(2);
      }

      // First Letter Achievement
      if (!currentAchievements.includes(3)) {
        const hasFirstLetter = await this.checkFirstLetter(userID);
        if (hasFirstLetter) newAchievements.push(3);
      }

      // Bronze Achievement (25% completion)
      if (!currentAchievements.includes(4)) {
        const completion = this.calculateCompletion(currentAchievements.length, totalAchievements);
        if (completion >= 25) newAchievements.push(4);
      }

      // Learned The Alphabet Achievement
      if (!currentAchievements.includes(5)) {
        const knowsAlphabet = await this.checkAlphabetComplete(userID);
        if (knowsAlphabet) newAchievements.push(5);
      }

      // Silver Achievement (50% completion)
      if (!currentAchievements.includes(6)) {
        const completion = this.calculateCompletion(currentAchievements.length, totalAchievements);
        if (completion >= 50) newAchievements.push(6);
      }

      // Learned The Dictionary Achievement
      if (!currentAchievements.includes(7)) {
        const knowsDictionary = await this.checkDictionaryComplete(userID);
        if (knowsDictionary) newAchievements.push(7);
      }

      // Gold Achievement (75% completion)
      if (!currentAchievements.includes(8)) {
        const completion = this.calculateCompletion(currentAchievements.length, totalAchievements);
        if (completion >= 75) newAchievements.push(8);
      }

      // Platinum Achievement (100% completion)
      if (!currentAchievements.includes(9)) {
        const completion = this.calculateCompletion(currentAchievements.length, totalAchievements);
        if (completion >= 100) newAchievements.push(9);
      }

      // Update if new achievements found
      if (newAchievements.length > 0) {
        // Use manager to update instead of direct API call
        const success = await AchievementsManager.updateAchievements(
          [...currentAchievements, ...newAchievements]
        );
        
        if (!success) {
          console.error("Failed to persist new achievements");
        }
      }

      return newAchievements;
    } catch (error) {
      console.error('Achievement check failed:', error);
      return [];
    }
  }

  static calculateCompletion(completedCount, totalCount) {
    if (totalCount <= 0) return 0;
    return Math.round((completedCount / totalCount) * 100);
  }

  // Placeholder methods - implement based on your actual checks
  static async checkFirstAlphabet(userID) {
    // Implement actual check
    return false;
  }

  static async checkFirstLetter(userID) {
    // Implement actual check
    return false;
  }

  static async checkAlphabetComplete(userID) {
    // Implement actual check
    return false;
  }

  static async checkDictionaryComplete(userID) {
    // Implement actual check
    return false;
  }
}