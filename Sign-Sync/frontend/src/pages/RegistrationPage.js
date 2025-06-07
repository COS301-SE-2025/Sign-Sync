import React from "react";
import { Link } from "react-router-dom";

class RegistrationPage extends React.Component 
{
    constructor(props)
    {
        super(props);
        
        this.state = {
            email: '',
            password: '',
            confirmPassword: '',
            errors: {}
        };
    }

    validateForm = () => 
    {
        const { email, password, confirmPassword} = this.state;
        const errors = {};

        if(!email.trim()) 
        {
            errors.email = "Email is required.";
        } 
        else if(!/^[\w-.]+@([\w-]+\.)+[\w-]{2,4}$/.test(email)) 
        {
            errors.email = "Enter a valid email address.";
        }

        if(!password.trim()) 
        {
            errors.password = "Password is required.";
        } 
        else if(password.length < 6) 
        {
            errors.password = "Password must be at least 6 characters.";
        }

        if(!confirmPassword.trim()) 
        {
            errors.confirmPassword = "Please confirm your password.";
        } 
        else if(password !== confirmPassword) 
        {
            errors.confirmPassword = "Passwords do not match.";
        }

        this.setState({ errors });

        return Object.keys(errors).length === 0;
    };

    handleInputChange = (e) =>
    {        
        const { name, value } = e.target;
        this.setState({ [name]: value });
    };

    handleSubmit = async (e) => 
    {
        e.preventDefault();

        if(!this.validateForm()) return;

        const { email, password } = this.state;

        try
        {
            const response = await fetch('/userApi/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ email: email, password: password }),
            });

            if(response.ok)
            {
                await response.json();
                
                alert("Registration successful!, redirecting to login page...");

                window.location.href = '/login';
            }
            else
            {
                const errorData = await response.json();
                alert(`Registration failed: ${errorData.message}`);
                console.error("Registration error:", errorData);
            }
        }
        catch(error)
        {
            console.error("Error during registration:", error);
            alert("An error occurred during registration. Please try again.");
        }
    };

    render() 
    {
        const { email, password, confirmPassword} = this.state;

        return (
            <div className="flex relative justify-center items-center w-screen h-screen bg-blue-900">
                <div className="absolute inset-0 m-auto bg-sky-950 w-[90%] max-w-[1237px] h-full rounded-[55px] max-md:rounded-[40px] max-sm:rounded-[30px]" />

                <form
                    onSubmit={this.handleSubmit}
                    className="box-border flex relative flex-col gap-12 items-start p-12 bg-white rounded-xl border border-amber-50 border-solid max-h-[90vh] overflow-y-auto min-w-80 w-[773px] z-[2] max-md:gap-8 max-md:p-8 max-md:h-auto max-md:max-w-[600px] max-md:w-[90%] max-sm:gap-6 max-sm:p-6 max-sm:max-w-[400px] max-sm:w-[95%]"
                >
                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Email
                        </label>
                        <input
                            type="text"
                            placeholder="Enter email"
                            name="email"
                            value={email}
                            onChange={this.handleInputChange}
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                        />
                        {this.state.errors.email && (
                            <p className="text-red-600 text-lg">{this.state.errors.email}</p>
                        )}
                    </div>

                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Password
                        </label>
                        <input
                            type="password"
                            placeholder="Enter password"
                            name="password"
                            value={password}
                            onChange={this.handleInputChange}
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                        />
                        {this.state.errors.password && (
                            <p className="text-red-600 text-lg">{this.state.errors.password}</p>
                        )}
                    </div>

                    <div className="flex flex-col gap-4 items-start w-full">
                        <label className="self-stretch text-3xl font-bold leading-10 text-stone-900 max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Confirm Password
                        </label>
                        <input
                            type="password"
                            placeholder="Please confirm your password"
                            name="confirmPassword"
                            value={confirmPassword}
                            onChange={this.handleInputChange}
                            className="self-stretch px-8 py-6 text-2xl font-bold leading-4 bg-white rounded-lg border border-solid flex-[1_0_0] min-w-60 text-zinc-900 max-md:px-6 max-md:py-5 max-md:text-sm max-sm:px-5 max-sm:py-4 max-sm:text-sm"
                        />
                        {this.state.errors.confirmPassword && (
                            <p className="text-red-600 text-lg">{this.state.errors.confirmPassword}</p>
                        )}
                    </div>

                    <div className="flex gap-8 items-center self-stretch">
                        <button
                            type="submit"
                            className="flex justify-center items-center gap-4 p-6 text-3xl font-bold leading-4 bg-red-900 text-white rounded-lg cursor-pointer flex-[1_0_0] max-md:p-5 max-md:text-sm max-sm:p-4 max-sm:text-sm"
                        >
                            Register
                        </button>
                    </div>

                    <div className="w-full flex justify-center">
                        <div className="text-3xl font-bold tracking-normal leading-10 text-stone-900 text-center max-md:text-3xl max-md:leading-9 max-sm:text-2xl max-sm:leading-8">
                            Already have an account?
                            <Link
                                to="/login"
                                className="ml-2 text-2xl font-bold text-white bg-red-900 px-4 py-2 rounded max-sm:text-sm"
                            >
                                Sign In
                            </Link>
                        </div>
                    </div>
                </form>
            </div>
        );
    }
}

export default RegistrationPage;
