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
                <label className="block text-sm font-medium text-gray-700">
                    {label}
                </label>
                <div className="relative mt-2">
                    <select
                        value={value}
                        onChange={this.handleChange}
                        className="cursor-pointer flex items-center justify-between w-full px-4 py-2 border rounded bg-white border-gray-300"
                    >
                        {options.map((opt, index) => (
                            <option key={index} value={opt}>
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
