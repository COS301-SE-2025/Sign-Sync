import React from "react";

class SettingsRow extends React.Component 
{
    render() 
    {
        const { title, value, className = "" } = this.props;

        return (
            <div className={`flex border-b pb-3 ${className}`}>
                <h3 className="w-32 text-black font-medium">{title}</h3>
                <p className="flex-1 text-zinc-700">{value}</p>
            </div>
        );
    }
}

export default SettingsRow;
