import React from "react";
import SideNavbar from "../components/sideNavbar";
import SettingsRow from "../components/SettingsRow";
import SelectField from "../components/SelectField";
import SliderField from "../components/SliderField";
import PreferenceManager from "../components/PreferenceManager";

class SettingsPage extends React.Component 
{
    constructor(props) 
    {
        super(props);
        
        const prefs = PreferenceManager.getPreferences();

        this.state = {
            displayMode: prefs.displayMode || 'Light Mode',
            fontSize: prefs.fontSize || 'Medium',
            email: '',
            user: null
        };
    }

    async componentDidMount() 
    {
        const user = JSON.parse(localStorage.getItem('user'));

        if(!user) 
        {
           // window.location.href = '/login'; //if not logged in, login first to access settings.

           this.setState({user: null});
            return;
        }

        this.setState({ email: user.email, user});

        await PreferenceManager.initialize();
        const loadedPrefs = PreferenceManager.getPreferences();

        this.setState({
            displayMode: loadedPrefs.displayMode,
            fontSize: loadedPrefs.fontSize || 'Medium',
        });

        PreferenceManager.applyDisplayMode(loadedPrefs.displayMode);
    }

    handleSavePreferences = async () => 
    {
        const user = JSON.parse(localStorage.getItem('user'));

        const { displayMode, fontSize } = this.state;
        
        try 
        {
            const response = await fetch(`/userApi/preferences/${user.userID}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ displayMode, fontSize })
            });

            if(response.ok) 
            {
                alert("Preferences saved successfully.");
            } 
            else 
            {
                const data = await response.json();

                alert("Failed to save preferences: " + data.message);
            }
        } 
        catch(error) 
        {
            console.error("Error saving preferences:", error);
            alert("An error occurred while saving.");
        }
    };

    handleChange = (field, value) => 
    {
        this.setState({ [field]: value });
        PreferenceManager.updatePreferences({ [field]: value });

        if(field === "displayMode") 
        {
            PreferenceManager.applyDisplayMode(value);
        }
        else if (field === "fontSize") 
        {
            PreferenceManager.applyFontSize(value);
        }
    };

    handleDeleteAccount = async () => 
    {
        const confirmed = window.confirm("Are you sure you want to delete your account? This action cannot be undone.");

        if(!confirmed) return;

        const user = JSON.parse(localStorage.getItem('user'));

        try 
        {
            const response = await fetch(`/userApi/deleteAccount/${user.userID}`, {
                method: 'DELETE'
            });

            if(response.ok) 
            {
                localStorage.removeItem('user');
                window.location.href = '/';
            } 
            else 
            {
                const data = await response.json();
                alert("Failed to delete account: " + data.message);
            }
        } 
        catch(error) 
        {
            console.error("Error deleting account:", error);
            alert("An error occurred while deleting the account.");
        }
    };


    render() 
    {
        const {displayMode, fontSize, email, user} = this.state;
        const isDarkMode = displayMode === "Dark Mode";

        return (
            <section className="flex h-screen bg-white dark:bg-gray-900 text-black dark:text-white transition-colors duration-300">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Right: Main Settings */}
                <div className="flex-1 overflow-y-auto flex justify-center items-center px-20 pt-14 pb-14 max-md:px-5 max-md:pt-12">
                    
                    
                    {/* Blurred main content when no user */}
                    <div className={!user ? "blur-sm" : ""}>
                        <div className="w-full max-w-lg bg-white dark:bg-gray-800 p-10 rounded-xl shadow-md dark:shadow-lg transition-all duration-300">
                        
                            <SettingsRow title="Email:" value={email} className="mt-4" />

                            <div className="mt-12 space-y-7">
                                <SelectField
                                    label="Display mode"
                                    value={displayMode}
                                    onChange={(value) => this.handleChange("displayMode", value)}
                                    options={["Light Mode", "Dark Mode"]}
                                />

                                <SelectField
                                    label="Preferred Avatar"
                                    //value={this.state.preferredAvatar}
                                    onChange={(value) => this.handleChange("preferredAvatar", value)}
                                    options={["Default", "Custom1", "Custom2"]}
                                />

                                <SliderField
                                    leftLabel="Small"
                                    rightLabel="Large"
                                    description="Font Size"
                                    value={fontSize}
                                    onChange={(value) => this.handleChange("fontSize", value)}
                                />

                            <div className="mt-10 flex justify-between">
                                    <button
                                        onClick={this.handleSavePreferences}
                                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition-all duration-200"
                                    >
                                        Save Preferences
                                    </button>

                                    <button
                                        onClick={this.handleDeleteAccount}
                                        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600 transition-all duration-200"
                                    >
                                        Delete Account
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Login required overlay */}
                    {!user && (
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className={`p-8 rounded-lg shadow-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'} border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'} z-10 max-w-md text-center`}>
                                <h2 className="text-2xl font-bold mb-4">Login Required</h2>
                                <p className="mb-6">Please log in to view and change your settings</p>
                                <button
                                    onClick={() => window.location.href = '/login'}
                                    className={`px-6 py-2 rounded-lg ${isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-500 hover:bg-blue-600'} text-white font-medium transition-colors`}
                                >
                                    Go to Login
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </section>
        );
    }
}

export default SettingsPage;
