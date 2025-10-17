import AchievementsManager from "../components/AchievementsManager";

// Base = the 5 “source-of-truth” achievements
const BASE_IDS = [1, 2, 3, 5, 7]; // firstLogin, firstLetter, firstWord, allLetters, allWords

export default class AchievementChecker {
  static async checkAchievements(userID) {
    let newlyUnlocked = [];

    try {
      if (!AchievementsManager.userID) {
        await AchievementsManager.initialize();
      }

      // Start from whatever the manager currently thinks is unlocked
      const current = new Set(AchievementsManager.getAchievements() || []);

      // Read the latest booleans we cached from the server
      const b = AchievementsManager.rawBooleans || {};

      // Map booleans -> IDs (include firstLogin now)
      if (b.firstLogin)  current.add(1);
      if (b.firstLetter) current.add(2);
      if (b.firstWord)   current.add(3);
      if (b.allLetters)  current.add(5);
      if (b.allWords)    current.add(7);

      // Compute milestones from BASE only
      const baseUnlocked = BASE_IDS.filter(id => current.has(id)).length;
      const pct = Math.round((baseUnlocked / BASE_IDS.length) * 100);
      if (pct >= 25)  current.add(4);  // bronze
      if (pct >= 50)  current.add(6);  // silver
      if (pct >= 75)  current.add(8);  // gold
      if (pct >= 100) current.add(9);  // platinum

      // Persist if anything new was added
      const before = AchievementsManager.getAchievements() || [];
      const after  = Array.from(current);
      const diff   = after.filter(id => !before.includes(id));

      if (diff.length > 0) {
        // This will convert IDs -> booleans and store in Mongo,
        // then read back and normalize again.
        const ok = await AchievementsManager.updateAchievements(after);
        if (!ok) console.error("Failed to persist new achievements");
        newlyUnlocked = diff;
      }

      return newlyUnlocked;
    } catch (err) {
      console.error("Achievement check failed:", err);
      return [];
    }
  }
}
