class AchievementsManager {
  static achievements = [];   // always an array of IDs in memory
  static userID = null;
  static rawBooleans = null;  // last server boolean object

  // ---- FULL mapping (IDs <-> booleans) ----
  static idsToBooleans(ids = []) {
    return {
      firstLogin:  ids.includes(1),
      firstLetter: ids.includes(2),
      firstWord:   ids.includes(3),
      bronze:      ids.includes(4),
      allLetters:  ids.includes(5),
      silver:      ids.includes(6),
      allWords:    ids.includes(7),
      gold:        ids.includes(8),
      platinum:    ids.includes(9),
    };
  }

  static booleansToIds(b = {}) {
    const map = {
      firstLogin: 1,
      firstLetter: 2,
      firstWord: 3,
      bronze: 4,
      allLetters: 5,
      silver: 6,
      allWords: 7,
      gold: 8,
      platinum: 9,
    };
    return Object.keys(map).filter(k => !!b[k]).map(k => map[k]);
  }

  static get DEFAULTS() {
    // ensure every key exists even if missing in DB
    return this.idsToBooleans([]); // all false
  }

  // Parse ANY server shape into { ids, booleans }
  static parseServerPayload(payload) {
    // unwrap { achievements: ... } if present
    const data = (payload && typeof payload === 'object' && 'achievements' in payload)
      ? payload.achievements
      : payload;

    if (Array.isArray(data)) {
      const ids = Array.from(new Set(data));
      return { ids, booleans: this.idsToBooleans(ids) };
    }

    // assume boolean object
    const booleans = { ...this.DEFAULTS, ...(data || {}) };
    return { ids: this.booleansToIds(booleans), booleans };
  }

  static async initialize() {
    const user = JSON.parse(localStorage.getItem('user'));
    if (!user || !user.userID) {
      console.warn("No user logged in - achievements not loaded");
      return false;
    }
    this.userID = user.userID;

    try {
      const resp = await fetch(`/userApi/achievements/${this.userID}`);
      if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);

      const raw = await resp.json();
      const { ids, booleans } = this.parseServerPayload(raw);

      this.achievements = ids;
      this.rawBooleans  = booleans;
      return true;
    } catch (err) {
      console.error("Failed to load achievements:", err);
      this.achievements = [];
      this.rawBooleans  = { ...this.DEFAULTS };
      return false;
    }
  }

  static getAchievements() {
    return Array.isArray(this.achievements) ? this.achievements : [];
  }

  /**
   * Persist an updated array of achievement IDs.
   * We WRITE booleans (matches Mongo schema) and accept any response shape.
   */
  static async updateAchievements(newIdsArray) {
    if (!this.userID) {
      console.error("No user ID available");
      return false;
    }
    try {
      const deduped = Array.from(new Set(newIdsArray || []));
      const booleansPayload = this.idsToBooleans(deduped);

      const response = await fetch(`/userApi/achievements/${this.userID}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(booleansPayload)   // write booleans to Mongo
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const raw = await response.json();
      const { ids, booleans } = this.parseServerPayload(raw);

      this.achievements = ids;
      this.rawBooleans  = booleans;
      return true;
    } catch (error) {
      console.error("Update failed:", error);
      return false;
    }
  }

  /**
   * Add new IDs (if not present) and persist.
   */
  static async addAchievements(idsToAdd = []) {
    const set = new Set([...(this.getAchievements()), ...idsToAdd]);
    return await this.updateAchievements(Array.from(set));
  }
}

export default AchievementsManager;
