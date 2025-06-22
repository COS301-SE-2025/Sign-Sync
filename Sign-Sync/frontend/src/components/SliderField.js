import React from "react";

const FONT_SIZE_OPTIONS = ["Small", "Medium", "Large"];

class SliderField extends React.Component 
{
    getIndexFromValue = (value) => 
    {
        const index = FONT_SIZE_OPTIONS.indexOf(value);
        return index !== -1 ? index : 1;
    };

    handleChange = (e) => 
    {
        const index = parseInt(e.target.value, 10);
        const sizeLabel = FONT_SIZE_OPTIONS[index];

        if(this.props.onChange) 
        {
            this.props.onChange(sizeLabel);
        }
    };

    render() 
    {
        const { leftLabel, rightLabel, description, className = "", value } = this.props;

        // Convert string "Small", "Medium", "Large" to 0/1/2
        const currentIndex = this.getIndexFromValue(value);

        return (
            <div className={`w-full ${className}`}>
                <div className="flex justify-between text-sm text-gray-700 dark:text-gray-300">
                    <span>{leftLabel}</span>
                    <span>{rightLabel}</span>
                </div>

                <input
                    type="range"
                    min="0"
                    max="2"
                    step="1"
                    value={currentIndex}
                    onChange={this.handleChange}
                    className="w-full mt-2 accent-blue-600 dark:accent-blue-400"
                />

                {/* Show current label */}
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {description}: <strong>{FONT_SIZE_OPTIONS[currentIndex]}</strong>
                </p>
            </div>
        );
    }
}

export default SliderField;
