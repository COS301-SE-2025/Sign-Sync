import React from "react";
import SideNavbar from "../components/sideNavbar";
import SettingsRow from "../components/SettingsRow";
import SelectField from "../components/SelectField";
import SliderField from "../components/SliderField";

class SettingsPage extends React.Component 
{
    constructor(props) 
    {
        super(props);
        
        this.state = {
            displayMode: "Light Mode",
            preferredAvatar: "Default"
        };
    }

    componentDidMount() 
    {
        const user = JSON.parse(localStorage.getItem('user'));

        if(!user) 
        {
            window.location.href = '/login'; //if not logged in, login first to access settings.
            return;
        }

        fetch(`/user/preferences/${user.userID}`)
            .then(res => res.json())
            .then(data => 
            {
                if(data.preferences) 
                {
                    this.setState(data.preferences);

                    // Apply dark mode
                    if(data.preferences.displayMode === "Dark Mode") 
                    {
                        document.documentElement.classList.add('dark');
                    } 
                    else 
                    {
                        document.documentElement.classList.remove('dark');
                    }
                }
            })
            .catch(err => console.error('Error loading preferences:', err));
    }

    handleSavePreferences = async () => 
    {
        const user = JSON.parse(localStorage.getItem('user'));
        const { displayMode, preferredAvatar } = this.state;

        try 
        {
            const response = await fetch(`/user/preferences/${user.userID}`, {
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
        this.setState({ [field]: newValue });
    };

    render() 
    {
        return (
            <section className="flex h-screen overflow-hidden bg-white">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Right: Main Settings */}
                <div className="flex-1 flex justify-center px-20 pt-14 pb-14 max-md:px-5 max-md:pt-12">
                    <div className="w-full max-w-lg">
                        <SettingsRow 
                            title="Name" 
                            value="Apollo Projects" 
                        />
                        
                        <SettingsRow 
                            title="Email" 
                            value="apolloprojects.cos301@gmail.com" 
                            className="mt-8" 
                        />

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
                                options={["Default", "Custom1", "Custom2",]}
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
                                className="mt-10 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
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
