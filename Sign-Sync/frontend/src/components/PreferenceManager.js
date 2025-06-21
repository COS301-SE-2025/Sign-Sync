class PreferenceManager 
{
    static preferences = {
        displayMode: 'Light Mode',
        preferredAvatar: 'Default',
    };

    static async initialize() 
    {
        const user = JSON.parse(localStorage.getItem('user'));
        
        if(!user) return;

        try 
        {
            const response = await fetch(`/userApi/preferences/${user.userID}`);
            const data = await response.json();

            if(data.preferences) 
            {
                PreferenceManager.preferences = data.preferences;
                localStorage.setItem('preferences', JSON.stringify(data.preferences));
                PreferenceManager.applyDisplayMode(data.preferences.displayMode);
            }
        } 
        catch(err) 
        {
            console.error("Failed to load preferences:", err);
        }
    }

    static getPreferences() 
    {
        return PreferenceManager.preferences;
    }

    static updatePreferences(newPrefs) 
    {
        PreferenceManager.preferences = {
            ...PreferenceManager.preferences,
            ...newPrefs,
        };

        localStorage.setItem('preferences', JSON.stringify(PreferenceManager.preferences));
    }

    static applyDisplayMode(mode) 
    {
        if(mode === 'Dark Mode') 
        {
            document.documentElement.classList.add('dark');
        } 
        else 
        {
            document.documentElement.classList.remove('dark');
        }
    }
}

export default PreferenceManager;
