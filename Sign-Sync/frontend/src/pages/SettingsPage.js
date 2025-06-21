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
        
        const prefs = JSON.parse(localStorage.getItem('preferences')) || PreferenceManager.getPreferences();

        this.state = {
            displayMode: prefs.displayMode,
            preferredAvatar: prefs.preferredAvatar,
        };
    }

    async componentDidMount() 
    {
        const user = JSON.parse(localStorage.getItem('user'));

        if(!user) 
        {
            window.location.href = '/login'; //if not logged in, login first to access settings.
            return;
        }

        this.setState({ email: user.email });

        await PreferenceManager.initialize();
        const loadedPrefs = PreferenceManager.getPreferences();

        this.setState({
            displayMode: loadedPrefs.displayMode,
            preferredAvatar: loadedPrefs.preferredAvatar,
        });

        PreferenceManager.applyDisplayMode(loadedPrefs.displayMode);
    }

    handleSavePreferences = async () => 
    {
        const user = JSON.parse(localStorage.getItem('user'));
        const { displayMode, preferredAvatar } = this.state;

        try 
        {
            const response = await fetch(`/userApi/preferences/${user.userID}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ displayMode, preferredAvatar })
            });

            const data = await response.json();

            if(response.ok) 
            {
                alert("Preferences saved successfully.");
            } 
            else 
            {
                alert("Failed to save preferences: " + data.message);
            }
        } 
        catch(error) 
        {
            console.error("Error saving preferences:", error);
            alert("An error occurred while saving.");
        }
    };

    handleChange = (field, newValue) => 
    {
        this.setState(
            { [field]: newValue },
            () => {
                PreferenceManager.updatePreferences({
                    displayMode: this.state.displayMode,
                    preferredAvatar: this.state.preferredAvatar,
                });

                if(field === "displayMode") 
                {
                    PreferenceManager.applyDisplayMode(newValue);
                }
            }
        );
    };

    render() 
    {
        return (
            <section className="flex h-screen overflow-hidden bg-white dark:bg-gray-900 text-black dark:text-white transition-colors duration-300">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Right: Main Settings */}
                <div className="flex-1 flex justify-center px-20 pt-14 pb-14 max-md:px-5 max-md:pt-12">
                    <div className="w-full max-w-lg bg-white dark:bg-gray-800 p-10 rounded-xl shadow-md dark:shadow-lg transition-all duration-300">
                        
                        <SettingsRow title="Email" value={this.state.email} className="mt-4" />

                        <div className="mt-12 space-y-7">
                            <SelectField
                                label="Display mode"
                                value={this.state.displayMode}
                                onChange={(value) => this.handleChange("displayMode", value)}
                                options={["Light Mode", "Dark Mode"]}
                            />

                            <SelectField
                                label="Preferred Avatar"
                                value={this.state.preferredAvatar}
                                onChange={(value) => this.handleChange("preferredAvatar", value)}
                                options={["Default", "Custom1", "Custom2"]}
                            />

                            <SliderField
                                leftLabel="Small"
                                rightLabel="Large"
                                description="Font Size"
                            />

                            <SliderField
                                leftLabel="Speed"
                                rightLabel="Accuracy"
                                description="Performance of Application"
                            />

                            <SliderField
                                leftLabel="Slow"
                                rightLabel="Fast"
                                description="Speed of AI Speech"
                            />

                            <button
                                onClick={this.handleSavePreferences}
                                className="mt-10 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition-all duration-200"
                            >
                                Save Preferences
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}

export default SettingsPage;
