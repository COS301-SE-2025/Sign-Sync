import React from "react";

class SelectField extends React.Component 
{
    handleChange = (event) => 
    {
        const { onChange } = this.props;
        
        if(onChange) 
        {
            onChange(event.target.value);
        }
    };

    render() 
    {
        const { label, value, className = "", options = [] } = this.props;

        return (
            <div className={`w-full ${className}`}>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    {label}
                </label>
                <div className="relative">
                    <select
                        value={value}
                        onChange={this.handleChange}
                        className="cursor-pointer w-full px-4 py-2 border rounded 
                                   bg-white text-black border-gray-300 
                                   dark:bg-gray-700 dark:text-white dark:border-gray-600
                                   focus:outline-none focus:ring-2 focus:ring-blue-500
                                   transition-colors duration-200"
                    >
                        {options.map((opt, index) => (
                            <option 
                                key={index} 
                                value={opt} 
                                className="bg-white dark:bg-gray-700 text-black dark:text-white"
                            >
                                {opt}
                            </option>
                        ))}
                    </select>
                </div>
            </div>
        );
    }
}

export default SelectField;
