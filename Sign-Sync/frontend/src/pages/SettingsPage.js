import React from "react";
import SideNavbar from "../components/sideNavbar";
import SettingsRow from "../components/SettingsRow";
import SelectField from "../components/SelectField";
import SliderField from "../components/SliderField";
import PreferenceManager from "../components/PreferenceManager";
import { toast } from "react-toastify";

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
            user: null,
            preferredAvatar : prefs.preferredAvatar || 'Zac',
            animationSpeed: prefs.animationSpeed || 1,
            speechSpeed: prefs.speechSpeed || 1,
            speechVoice: prefs.speechVoice || 'George',
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

        const { displayMode, fontSize, preferredAvatar, animationSpeed, speechSpeed, speechVoice } = this.state;
        
        try 
        {
            const response = await fetch(`/userApi/preferences/${user.userID}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ displayMode, fontSize, preferredAvatar, animationSpeed, speechSpeed, speechVoice})
            });
            if(response.ok) 
            {
                toast.success("Preferences saved successfully.");
            } 
            else 
            {
                const data = await response.json();

                toast.error("Failed to save preferences: " + data.message);
            }
        } 
        catch(error) 
        {
            console.error("Error saving preferences:", error);
            toast.error("An error occurred while saving.");
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

    toastConfirm = (message) => 
    {
        return new Promise((resolve) => 
        {
            const id = toast.info(
                <div>
                    {message}
                    <div className="mt-2 flex justify-end gap-2">
                        <button
                            onClick={() => { toast.dismiss(id); resolve(true); }}
                            className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                        >
                            Yes
                        </button>
                        <button
                            onClick={() => { toast.dismiss(id); resolve(false); }}
                            className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
                        >
                            No
                        </button>
                    </div>
                </div>,
                { autoClose: false, closeOnClick: false, draggable: false }
            );
        });
    };

    handleDeleteAccount = async () => 
    {
        const confirmed = await this.toastConfirm("Are you sure you want to delete your account? This action cannot be undone.");

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
                toast.success("Account deleted successfully, redirecting to splash Page");
                
                setTimeout(() => { window.location.href = '/' }, 1200);
            } 
            else 
            {
                const data = await response.json();
                toast.error("Failed to delete account: " + data.message);
            }
        } 
        catch(error) 
        {
            console.error("Error deleting account:", error);
            toast.error("An error occurred while deleting the account.");
        }
    };


    render() 
    {
        const isDarkMode = displayMode === "Dark Mode";
        const {displayMode, fontSize, email, preferredAvatar, animationSpeed, speechSpeed, speechVoice, user} = this.state;

        return (
            <section className="flex h-screen dark:bg-gray-900 text-black dark:text-white transition-colors duration-300" style={{ background: isDarkMode 
                                                                                                                                    ? "linear-gradient(135deg, #0a1a2f 0%, #14365c 60%, #5c1b1b 100%)"
                                                                                                                                    : 'linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)'}}>
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Right: Main Settings */}
                <div className="flex-1 overflow-y-auto relative flex justify-center items-center px-20 pt-14 pb-14 max-md:px-5 max-md:pt-12">
                    
                    
                    {/* Blurred main content when no user */}
                    <div className={!user ? "blur-sm" : ""}>
                        <div className="w-full max-w-lg bg-white dark:bg-gray-800 p-10 rounded-xl shadow-md dark:shadow-lg transition-all duration-300">
                        
                        <SettingsRow title="Email" value={email} className="mt-4" />

                        <div className="mt-12 space-y-7">
                            <SelectField
                                label="Display mode"
                                value={displayMode}
                                onChange={(value) => this.handleChange("displayMode", value)}
                                options={["Light Mode", "Dark Mode"]}
                            />

                            <SelectField
                                label="Preferred Avatar"
                                value={preferredAvatar}
                                onChange={(value) => this.handleChange("preferredAvatar", value)}
                                options={["Zac", "Jenny"]}
                            />

                            <SliderField
                                leftLabel="Small"
                                rightLabel="Large"
                                description="Font Size"
                                value={fontSize}
                                OPTIONS={["Small", "Medium","Large"]}
                                onChange={(value) => this.handleChange("fontSize", value)}
                            />

                            <SliderField
                                leftLabel="Slow"
                                rightLabel="Fast"
                                description="Animation Speed"
                                value={animationSpeed}
                                OPTIONS={["Very Slow", "Slow","Normal","Fast","Very Fast"]}
                                onChange={(value) => this.handleChange("animationSpeed", value)}
                            />

                            <SelectField
                                label="Preferred Voice"
                                value={speechVoice}
                                onChange={(value) => this.handleChange("speechVoice", value)}
                                options={["George", "Hazel","Susan"]}
                            />

                            <SliderField
                                leftLabel="Slow"
                                rightLabel="Fast"
                                description="Speech Speed"
                                value={speechSpeed}
                                OPTIONS={["Slow","Normal","Fast"]}
                                onChange={(value) => this.handleChange("speechSpeed", value)}
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
