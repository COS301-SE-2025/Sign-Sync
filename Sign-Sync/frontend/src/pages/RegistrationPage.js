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
            <div
                className="flex items-center justify-center min-h-screen p-4"
                style={{
                background:
                    "linear-gradient(135deg, #102a46, #1c4a7c, #d32f2f)",
                }}
            >
                <form
                    onSubmit={this.handleSubmit}
                    className="w-full max-w-md bg-white rounded-2xl shadow-2xl p-8 space-y-6"
                >
                    <h2 className="text-2xl font-extrabold text-center text-gray-900">
                        Register
                    </h2>

                    <div className="flex flex-col">
                        <label
                            htmlFor="email"
                            className="text-gray-700 font-medium mb-2"
                        >
                            Email
                        </label>

                        <input
                            type="text"
                            id="email"
                            name="email"
                            value={email}
                            onChange={this.handleInputChange}
                            placeholder="Enter your email"
                            className={`px-4 py-3 rounded-lg border ${
                                errors.email ? "border-red-500" : "border-gray-300"
                            } focus:outline-none focus:ring-2 focus:ring-red-600`}
                        />

                        {errors.email && (
                        <p className="text-red-500 mt-1 text-sm">{errors.email}</p>
                        )}

                    </div>

                    <div className="flex flex-col">
                        <label
                            htmlFor="password"
                            className="text-gray-700 font-medium mb-2"
                        >
                            Password
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

                    <div className="flex flex-col">
                        <label
                            htmlFor="confirmPassword"
                            className="text-gray-700 font-medium mb-2"
                        >
                            Confirm Password
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

                    <button
                        type="submit"
                        className="w-full py-3 rounded-lg font-bold text-white transition-colors"
                        style={{
                        background:
                            "linear-gradient(to right, #102a46, #1c4a7c, #d32f2f)",
                        }}
                    >
                        Register
                    </button>

                    <p className="text-center text-gray-700">
                        Already have an account?{" "}
                        <Link
                            to="/login"
                            className="text-red-600 font-semibold hover:underline"
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
