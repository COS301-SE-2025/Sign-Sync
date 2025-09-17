import React from "react";
import { Link } from "react-router-dom";
import { toast } from "react-toastify";

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
                
                toast.success("Registration successful!, redirecting to login page...");

                setTimeout(() => { window.location.href = '/login' }, 1200);
            }
            else
            {
                const errorData = await response.json();
                toast.error(`Registration failed: ${errorData.message}`);
                console.error("Registration error:", errorData);
            }
        }
        catch(error)
        {
            console.error("Error during registration:", error);
            toast.error("An error occurred during registration. Please try again.");
        }
    };

    render() 
    {
        const { email, password, confirmPassword, errors } = this.state;

        return (
            <div className="flex items-center justify-center min-h-screen p-4" style={{ background: "linear-gradient(135deg, #080C1A, #172034)"}} >
                <form
                    onSubmit={this.handleSubmit}
                    className="w-full max-w-2xl rounded-2xl shadow-2xl p-8 space-y-6" style={{ background: "#1B2432"}}
                >
                    <h2 className="text-2xl font-extrabold text-center text-white">
                        Create an account
                    </h2>

                    <p className="text-center text-white">
                        Please provide the following information
                    </p>
    
                    <div className="flex flex-col w-full mt-4">
                        <label htmlFor="email" className="text-white font-medium mb-2">
                            Email <span className="text-red-500">*</span>
                        </label>

                        <input
                            type="text"
                            id="email"
                            name="email"
                            value={email}
                            onChange={this.handleInputChange}
                            placeholder="Enter your email"
                            className={`px-4 py-3 rounded-lg border ${
                                errors.email ? "border-red-500" : "border-white"
                            } focus:outline-none focus:ring-2 focus:ring-red-600`}
                        />

                        {errors.email && (
                        <p className="text-red-500 mt-1 text-sm">{errors.email}</p>
                        )}

                    </div>

                    <div className="flex w-full gap-2">
                        <div className="flex flex-col flex-1">
                            <label htmlFor="password" className="text-white font-medium mb-2">
                                Password <span className="text-red-500">*</span>
                            </label>
                            
                            <input
                                type="password"
                                id="password"
                                name="password"
                                value={password}
                                onChange={this.handleInputChange}
                                placeholder="Enter your password"
                                className={`px-4 py-3 rounded-lg border ${
                                    errors.password ? "border-red-500" : "border-gray-300"
                                } focus:outline-none focus:ring-2 focus:ring-red-600`}
                            />

                            {errors.password && (
                            <p className="text-red-500 mt-1 text-sm">{errors.password}</p>
                            )}

                        </div>

                        <div className="flex flex-col flex-1">
                            <label htmlFor="confirmPassword" className="text-white font-medium mb-2">
                                Confirm Password <span className="text-red-500">*</span>
                            </label>
                            
                            <input
                                type="password"
                                id="confirmPassword"
                                name="confirmPassword"
                                value={confirmPassword}
                                onChange={this.handleInputChange}
                                placeholder="Confirm your password"
                                className={`px-4 py-3 rounded-lg border ${
                                    errors.confirmPassword
                                    ? "border-red-500"
                                    : "border-gray-300"
                                } focus:outline-none focus:ring-2 focus:ring-red-600`}
                            />
                            
                            {errors.confirmPassword && (
                            <p className="text-red-500 mt-1 text-sm">
                                {errors.confirmPassword}
                            </p>
                            )}

                        </div>
                    </div>
                    

                    <button
                        type="submit"
                        className="w-full py-3 rounded-lg font-bold text-white transition-colors bg-blue-600 hover:bg-blue-700 mt-4"
                    >
                        Register
                    </button>

                    <p className="text-center text-white">
                        Already have an account?{" "}
                        <Link
                            to="/login"
                            className="text-blue-600 font-semibold hover:underline"
                        >
                            Sign In
                        </Link>
                    </p>
                </form>
            </div>
        );
    }
}

export default RegistrationPage;
