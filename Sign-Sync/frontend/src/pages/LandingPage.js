import React from 'react';
import { Link } from "react-router-dom";

class LandingPage extends React.Component 
{
    render() 
    {
        return (
            <div className="landing-page">
                <h1>Welcome to Sign-Sync</h1>
                
                <p>
                    Sign-Sync is a real-time sign language translation platform that bridges the gap 
                    between spoken and signed communication. Whether you're learning, teaching, or 
                    communicating across language barriers, Sign-Sync makes signing accessible and efficient.
                </p>

                <p>
                    You can get started right away, or log in to access your saved preferences and a personalized experience.
                </p>

                <div>
                    <Link to="/login" className="btn btn-primary">Login for Preferences</Link>
                    <Link to="/translator" className="btn btn-secondary">Continue Without Logging In</Link>
                </div>
            </div>
        );
    }
}

export default LandingPage;
