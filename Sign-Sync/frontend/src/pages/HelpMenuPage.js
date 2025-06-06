import React from "react";
import SideNavbar from "../components/sideNavbar";

class HelpMenuPage extends React.Component 
{
    render()
    {
        return(
            <section className="flex h-screen overflow-hidden bg-white">
                {/* Left: Sidebar */}
                <div>
                    <SideNavbar />
                </div>

            </section>

        );
    }
}

export default HelpMenuPage;
