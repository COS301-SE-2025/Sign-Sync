import React from "react";
import { Link } from "react-router-dom";
import { toast } from "react-toastify";

class LoginPage extends React.Component 
{
    constructor(props)
    {
        super(props);
        
        this.state = {
            email: '',
            password: '',
            errors: {}
        };
    }

    validateForm = () => 
    {
        const { email, password } = this.state;
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
            const response = await fetch('/userApi/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ email: email, password: password }),
            });

            if(response.ok)
            {
                const data = await response.json();
                
                localStorage.setItem('user', JSON.stringify(data.user));

                toast.success("Login successful! Redirecting to Translator page...");

                setTimeout(() => { window.location.href = '/translator'; }, 1200);
            }
            else
            {
                const errorData = await response.json();

                toast.error(`Login failed: ${errorData.message}`);

                console.error("Login error:", errorData);
            }
        }
        catch(error)
        {
            console.error("Error during Login:", error);

            toast.error("An error occurred during Login. Please try again.");

        }
    };

    render() 
    {
        const { email, password, errors } = this.state;

        return (
            <div 
                className="flex items-center justify-center min-h-screen p-4" 
                style={{ background: "linear-gradient(to bottom, #080C1A, #172034)" }}
            >
                <form
                    onSubmit={this.handleSubmit}
                    className="w-full max-w-2xl rounded-2xl shadow-2xl p-8 space-y-6"
                    style={{ background: "#1B2432" }}
                >
                    <h2 className="text-2xl font-extrabold text-center text-white">
                        Login
                    </h2>

                    <p className="text-center text-white">
                        Please provide the following information
                    </p>

                    {/* Email */}
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

                    {/* Password */}
                    <div className="flex flex-col w-full mt-4">
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

                    {/* Login button */}
                    <button
                        type="submit"
                        className="w-full py-3 rounded-lg font-bold text-white transition-colors bg-blue-600 hover:bg-blue-700 mt-4"
                    >
                        Login
                    </button>

                    {/* Link to register */}
                    <p className="text-center text-white">
                        Don’t have an account?{" "}
                        <Link
                            to="/register"
                            className="text-blue-600 font-semibold hover:underline"
                        >
                            Sign Up
                        </Link>
                    </p>
                </form>
            </div>
        );
    }

}

export default LoginPage;


