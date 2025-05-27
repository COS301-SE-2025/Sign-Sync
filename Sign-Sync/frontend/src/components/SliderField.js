import React from "react";

class SliderField extends React.Component 
{
    constructor(props) 
    {
        super(props);

        this.state = {
            value: props.initialValue ?? 50,
        };
    }

    handleChange = (e) => 
    {
        const newValue = parseInt(e.target.value, 10);

        this.setState({ value: newValue });
        
        if(this.props.onChange)
        {
            this.props.onChange(newValue);
        }
    };

    render() 
    {
        const { leftLabel, rightLabel, description } = this.props;
        const { value } = this.state;

        return (
            <div>
                <div>
                    <span>{leftLabel}</span>
                    <span>{rightLabel}</span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={value}
                    onChange={this.handleChange}
                />
                <p>{description}</p>
            </div>
        );
    }
}

export default SliderField;