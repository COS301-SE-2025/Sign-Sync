import React from "react";
import Camera from "../components/Camera";
import TextBox from "../components/TextBox";
import SideNavbar from "../components/sideNavbar";

class LoginPage extends React.Component
{
    render()
    {
        return (
            <div>
                <Camera />
                <div>
                    <TextBox />
                </div>
            </div>
        )
    }
}

export default LoginPage;
