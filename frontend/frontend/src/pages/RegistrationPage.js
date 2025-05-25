import React from "react";

class RegistrationPage extends React.Component 
{
    render() 
    {
        return (
            <div>
                <div>
                    <label>Username</label>
                    <input type="text" placeholder="Enter username"></input>
                </div>
                <div>
                    <label>Email</label>
                    <input type="email" placeholder="Enter email"></input>
                </div>
                <div>
                    <label>Password</label>
                    <input type="password" placeholder="Enter password"></input>
                </div>
                <div>
                    <button>Register</button>
                </div>
                <div>
                    <div>
                        Already have an account?
                        <button>Sign In</button>
                    </div>
                </div>
            </div>
        );
    }
}

export default RegistrationPage;