import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

import handBtn from '../assets/hand.png';
import EducationBtn from "../assets/Education-icon.png";
import AchievementsBtn from "../assets/achievements.png";
import PractiseBtn from "../assets/homework.png"
 
class LandingPage extends React.Component 
{
  componentDidMount() 
  {
    localStorage.clear(); //clear any existing user data
  };

  render() 
  {
    return (
      <div
        className="flex flex-col items-center justify-center min-h-screen"
        style={{ background: "linear-gradient(135deg, #080C1A, #172034)" }}
      >
        <motion.div
          initial={{ opacity: 0, y: 150 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className="bg-[#0f172a]/70 backdrop-blur-xl border border-white/10 rounded-2xl p-8 sm:p-12 shadow-2xl max-w-md w-full text-center"
        >
          {/* Logo + Title */}
          <h1 className="text-3xl sm:text-4xl font-extrabold mb-6 text-white">
            <span className="flex justify-center items-center gap-2 mb-2">
              <img src={handBtn} alt="Hand Icon" className="w-10 h-10" />
              <span className="bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                Sign-Sync
              </span>
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-gray-300 text-sm sm:text-base mb-6">
            <span className="font-bold">Sign-sync</span> is a real time sign language
            translation platform that bridges the gap between spoken and signed
            communication.
          </p>

          {/* Feature list */}
          <div className="text-gray-200 text-left space-y-3 mb-8">
            <div className="flex items-center gap-3 bg-white/5 px-4 py-2 rounded-lg">
              <img src={EducationBtn} alt="Learn Icon" className="w-6 h-6" />
              <span>Learn sign words and alphabet</span>
            </div>
            <div className="flex items-center gap-3 bg-white/5 px-4 py-2 rounded-lg">
              <img src={AchievementsBtn} alt="Achievements Icon" className="w-6 h-6" />
              <span>Unlock achievements to track your progress</span>
            </div>
            <div className="flex items-center gap-3 bg-white/5 px-4 py-2 rounded-lg">
              <img src={PractiseBtn} alt="Practise Icon" className="w-6 h-6" />
              <span>Practice sign words and alphabet</span>
            </div>
          </div>

          {/* Bottom buttons */}
          <div className="flex flex-col sm:flex-row gap-4 mt-6">
            <Link
              to="/login"
              className="flex-1 border border-white text-white font-semibold py-3 rounded-lg hover:bg-white/10 transition-all"
            >
              Login
            </Link>
            <Link
              to="/translator"
              className="flex-1 bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition-all"
            >
              Continue to app
            </Link>
          </div>
        </motion.div>
      </div>
    );
  }

}

export default LandingPage;

