class SessionManager {
    static session = {
        hasSignedFirstLetter: false,
        hasCompletedAlphabet: false
    };

    static setSessionData(data) {
        this.session = { ...this.session, ...data };
    // Optional: persist to localStorage
        localStorage.setItem('signLanguageSession', JSON.stringify(this.session));
    }

    static getSessionData() {
        // Check localStorage first
        const storedSession = localStorage.getItem('signLanguageSession');
        if (storedSession) {
            this.session = JSON.parse(storedSession);
        }
        return this.session;
    }

    static clearSession() {
        this.session = {
            hasSignedFirstLetter: false,
            hasCompletedAlphabet: false
        };
        localStorage.removeItem('signLanguageSession');
    }
}

export const setSessionData = SessionManager.setSessionData.bind(SessionManager);
export const getSessionData = SessionManager.getSessionData.bind(SessionManager);
export const clearSession = SessionManager.clearSession.bind(SessionManager);