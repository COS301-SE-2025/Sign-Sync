import React from "react";

class SettingsRow extends React.Component 
{
    render() 
    {
        const { title, value } = this.props;

        return (
            <div>
                <h3>{title}</h3>
                <p>{value}</p>
            </div>
        );
    }
}

export default SettingsRow;