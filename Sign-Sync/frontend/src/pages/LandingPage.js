import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

class LandingPage extends React.Component 
{
  componentDidMount() 
  {
      localStorage.clear(); // Clear any existing user data
  };

  render() 
  {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-blue-900 text-white px-6">
        
        <motion.div
          initial={{ opacity: 0, y: 250 }}
          animate={{ opacity: 1, y: 0}}
          transition={{ duration: 1.5, ease: 'easeOut' }}
          className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-10 shadow-2xl max-w-3xl w-full text-center"
        >

          <h1 className="text-5xl font-bold mb-8">Welcome to Sign-Sync</h1>

          <div className="mb-10 space-y-4 text-lg leading-relaxed">
            <p>
              Sign-Sync is a real-time sign language translation platform that bridges the gap
              between spoken and signed communication. Whether you're learning, teaching, or
              communicating across language barriers, Sign-Sync makes signing accessible and efficient.
            </p>

            <p>
              You can get started right away, or log in to access your saved preferences and a personalized experience.
            </p>
          </div>

          <div className="flex gap-6 flex-col sm:flex-row justify-center">
            <Link
              to="/login"
              className="bg-white text-blue-900 font-bold py-3 px-6 rounded-lg hover:bg-gray-100 transition"
            >
              Login
            </Link>
            <Link
              to="/translator"
              className="bg-red-700 text-white font-bold py-3 px-6 rounded-lg hover:bg-red-800 transition"
            >
              Continue to App
            </Link>
          </div>
        </motion.div>
      </div>
    );
  }
}

export default LandingPage;
