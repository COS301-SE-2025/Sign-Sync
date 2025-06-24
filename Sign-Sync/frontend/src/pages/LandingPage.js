import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

class LandingPage extends React.Component 
{
  componentDidMount() 
  {
      localStorage.clear(); //clear any existing user data
  };

  render() 
  {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-900 to-indigo-900 text-white px-6">
        <motion.div
          initial={{ opacity: 0, y: 150 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          className="bg-white bg-opacity-10 backdrop-blur-xl border border-white border-opacity-20 rounded-3xl p-10 sm:p-14 shadow-2xl max-w-3xl w-full text-center animate-fade-in"
        >
          <h1 className="text-4xl sm:text-5xl font-light mb-2 text-white">
            Welcome to
          </h1>

          <h2 className="text-5xl sm:text-6xl font-extrabold mb-8 text-white drop-shadow-[0_0_10px_rgba(255,255,255,0.2)] whitespace-nowrap">
            Sign-Sync
          </h2>

          <div className="mb-10 space-y-5 text-lg sm:text-xl leading-relaxed text-gray-100">
            <p>
              Sign-Sync is a real-time sign language translation platform that bridges the gap
              between spoken and signed communication.
            </p>
            <p>
              Whether you're learning, teaching, or just exploring — we make signing more
              accessible, powerful, and intuitive.
            </p>
          </div>

          <div className="flex gap-6 flex-col sm:flex-row justify-center mt-6">
            <Link
              to="/login"
              className="bg-white text-blue-900 font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-blue-100 transition-all duration-300 transform hover:scale-105"
            >
              Login
            </Link>
            <Link
              to="/translator"
              className="bg-gradient-to-r from-indigo-500 to-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:from-indigo-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105"
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
