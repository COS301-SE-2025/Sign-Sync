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
                        </div>
                    </div>
                </div>
            </section>
        );
    }
}

export default SettingsPage;
