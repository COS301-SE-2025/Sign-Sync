import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

class LandingPage extends React.Component 
{
  render() 
  {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 2 }}
        className="flex flex-col items-center justify-center min-h-screen bg-blue-900 text-white px-6 text-center"
      >
        <h1 className="text-5xl font-bold mb-8">Welcome to Sign-Sync</h1>

        <div className="max-w-2xl mb-10 space-y-4 text-lg leading-relaxed">
          <p>
            Sign-Sync is a real-time sign language translation platform that bridges the gap
            between spoken and signed communication. Whether you're learning, teaching, or
            communicating across language barriers, Sign-Sync makes signing accessible and efficient.
          </p>

          <p>
            You can get started right away, or log in to access your saved preferences and a personalized experience.
          </p>
        </div>

        <div className="flex gap-6 flex-col sm:flex-row">
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
    );
  }
}

export default LandingPage;
