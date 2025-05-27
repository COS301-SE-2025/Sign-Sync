import React from "react";
import SideNavbar from "../components/sideNavbar";
import SettingsRow from "../components/SettingsRow";
import SelectField from "../components/SelectField";
import SliderField from "../components/SliderField";

class SettingsPage extends React.Component 
{
    constructor(props) {
        super(props);
        this.state = {
            displayMode: "Light Mode",
            preferredAvatar: "Default"
        };
    }

    handleChange = (field, newValue) => {
        this.setState({ [field]: newValue });
    };

    render() 
    {
        return (
            <section>
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

                {/* Right: Main Settings */}
                <div>
                    <div>
                        <SettingsRow 
                            title="Name" 
                            value="Apollo Projects" 
                        />
                        
                        <SettingsRow 
                            title="Email" 
                            value="apolloprojects.cos301@gmail.com" 
                        />

                        <div>
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
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}

export default SettingsPage;