import React from "react";

class LoginPage extends React.Component 
{
    render() 
    {
        return (
            <div>
                <div>
                    <label>Username</label>
                    <input type="text" placeholder="Enter username"/>
                </div>

                <div>
                    <label>Email</label>
                    <input type="email" placeholder="Enter email"/>
                </div>

                <div>
                    <label>Password</label>
                    <input type="password" placeholder="Enter password"/>
                </div>

                <div>
                    <button>Login</button>
                </div>

                <div>
                    <div>
                        Don't have an account?<button>Sign Up</button>
                    </div>
                </div>
            </div>
        );
    }
}

export default LoginPage;