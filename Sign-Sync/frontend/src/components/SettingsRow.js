import React from "react";

class SettingsRow extends React.Component 
{
    render() 
    {
        const { title, value, className = "" } = this.props;

        return (
            <div className={`flex border-b pb-3 border-gray-300 dark:border-gray-600 ${className}`}>
                <h3 className="w-32 text-black dark:text-white font-medium">{title}</h3>
                <p className="flex-1 text-zinc-700 dark:text-gray-300">{value}</p>
            </div>
        );
    }
}

export default SettingsRow;
