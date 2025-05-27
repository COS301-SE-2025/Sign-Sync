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
        const { label, value, options = [] } = this.props;

        return (
            <div>
                <label>
                    {label}
                </label>
                <div>
                    <select
                        value={value}
                        onChange={this.handleChange}
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