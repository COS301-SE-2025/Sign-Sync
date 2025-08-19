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
      <div className="flex flex-col items-center justify-center min-h-screen" style={{ background: "linear-gradient(135deg, #102a46 0%, #1c4a7c 60%, #d32f2f 100%)" }} >
              
        <motion.div
          initial={{ opacity: 0, y: 150 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          whileHover={{
            scale: 1.03,
            transition: { duration: 0.25, ease: 'easeOut' }
          }}
          className="bg-white bg-opacity-10 backdrop-blur-xl border border-white border-opacity-20 rounded-3xl p-10 sm:p-14 shadow-2xl max-w-3xl w-full text-center"
        >

          <h1 className="text-4xl sm:text-6xl font-extrabold mb-8 drop-shadow-md text-white">
            <span className="text-2xl sm:text-3xl text-white block mb-2">Welcome to</span>
            <span className="bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
              Sign-Sync
            </span>
          </h1>

          <div className="mb-10 space-y-5 text-lg sm:text-xl leading-relaxed text-gray-100">
            <p>
              Sign-Sync is a real-time sign language translation platform that bridges the gap
              between spoken and signed communication.
            </p>
            <p>
              Whether you're learning, teaching, or just exploring — we make signing more
              accessible, powerful, and intuitive.
            </p>
            <p>
              You can get started right away, or log in to access your saved preferences and a personalized experience.
            </p>
          </div>

          <div className="flex gap-6 flex-col sm:flex-row justify-center mt-6">
            <Link
              to="/login"
              className="bg-green-500 text-blue-900 font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-blue-100 transition-all duration-300 transform hover:scale-105"
            >
              Login
            </Link>
            <Link
              to="/translator"
              className="bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md hover:bg-blue-700 transition-all duration-300 transform hover:scale-105"
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

