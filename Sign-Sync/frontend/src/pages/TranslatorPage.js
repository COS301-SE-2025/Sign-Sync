import React from "react";
import Camera from "../components/Camera";
import SideNavbar from "../components/sideNavbar";
import textToSign from "../components/textToSign";
import TextToSign from "../components/textToSign";

class TranslatorPage extends React.Component
{
    render()
    {
        return (
            <div className={"flex flex-row"}>
                <SideNavbar/>
                <TextToSign/>
            </div>
        )
    }
}

export default TranslatorPage;
