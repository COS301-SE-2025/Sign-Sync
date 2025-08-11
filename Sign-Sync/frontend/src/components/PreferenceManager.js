class PreferenceManager 
{
    static preferences = {
        displayMode: 'Light Mode',
        preferredAvatar: 'Zac',
        fontSize: 'Medium'
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
                PreferenceManager.applyFontSize(data.preferences.fontSize);
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

    static applyFontSize(size) 
    {
        const root = document.documentElement;
        root.classList.remove('font-small', 'font-medium', 'font-large');

        switch(size) 
        {
            case "Small":
                root.classList.add('font-small');
                break;
            case "Medium":
                root.classList.add('font-medium');
                break;
            case "Large":
                root.classList.add('font-large');
                break;
            default:
                root.classList.add('font-medium');
        }
    }

}

export default PreferenceManager;
