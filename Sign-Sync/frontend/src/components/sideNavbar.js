import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { toast } from "react-toastify";
import { FaChevronDown } from "react-icons/fa";

import handLogo from "../assets/hand.png"
import translateBtn from "../assets/Translator-icon.png";
import EducationBtn from "../assets/Education-icon.png";
import PractiseBtn from "../assets/homework.png"
import AlphabetBtn from "../assets/abc-block.png";
import WordsBtn from "../assets/speech-bubble.png";
import AchievementsBtn from "../assets/achievements.png";
import SettingsBtn from "../assets/Settings-icon.png";
import HelpMenuBtn from "../assets/info-icon.png";
import SignoutBtn from "../assets/SignOut.png";
import SignInBtn from "../assets/SignIn.png"
import RegisterBtn from "../assets/Register.png";

const SideNavbar = () => {
  const location = useLocation();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [learnOpen, setLearnOpen] = useState(false);
  const [practiseOpen, setPractiseOpen] = useState(false);

  useEffect(() => {
    const user = localStorage.getItem("user");
    if (user) setIsLoggedIn(true);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    setIsLoggedIn(false);

    toast.success("Logout successful!, redirecting to Splash page...");
    setTimeout(() => {
      window.location.href = "/";
    }, 1200);
  };

  //helper: highlight active link
  const isActive = (path) =>
    location.pathname.startsWith(path)
      ? "bg-[#1a436b] text-white rounded"
      : "text-gray-200 hover:bg-[#1a436b] rounded";

  return (
    <div className="w-64 flex flex-col h-screen items-start px-0 pt-0 pb-5 bg-[#1B2432]">
      <div className="w-full flex flex-col gap-2 pt-6 text-white">
        {/* App title */}
        <div className="px-4 py-2 text-4xl font-bold flex items-center gap-2">
          <img src={handLogo} alt="Hand Logo" className="inline w-10 h-10" />
          <span className="bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent inline-block leading-normal">
            Sign-Sync
          </span>
        </div>

        {/* Translator */}
        <Link
          to="/translator"
          className={`px-4 py-2 text-2xl flex items-center gap-2 ${isActive(
            "/translator"
          )}`}
        >
          <img src={translateBtn} alt="Translator Icon" className="w-6 h-6" />
          Translator
        </Link>

        {/* Learn (expandable) */}
        <div
          className={`cursor-pointer flex items-center justify-between px-4 py-2 text-2xl ${isActive(
            "/learn"
          )}`}
          onClick={() => setLearnOpen(!learnOpen)}
        >
          <div className="flex items-center gap-2">
            <img src={EducationBtn} alt="Learn Icon" className="w-6 h-6" />
            <span>Learn</span>
          </div>
          <FaChevronDown
            className={`transition-transform duration-300 ${
              learnOpen ? "rotate-180" : ""
            }`}
          />
        </div>
        {learnOpen && (
          <div className="pl-8 flex flex-col gap-1 text-xl">
            <Link to="/learn-Alphabet" className={`${isActive("/learn-Alphabet")} px-2 py-1 flex items-center gap-2`}>
              <img src={AlphabetBtn} alt="Alphabet Icon" className="w-6 h-6 bg-white" />
              Alphabet
            </Link>
            <Link to="/learn-Words" className={`${isActive("/learn-Words")} px-2 py-1 flex items-center gap-2`}>
              <img src={WordsBtn} alt="Words Icon" className="w-6 h-6 bg-white" />
              Words
            </Link>
          </div>
        )}

        {/* Practise (expandable) */}
        <div
          className={`cursor-pointer flex items-center justify-between px-4 py-2 text-2xl ${isActive(
            "/practise"
          )}`}
          onClick={() => setPractiseOpen(!practiseOpen)}
        >
          <div className="flex items-center gap-2">
            <img src={PractiseBtn} alt="Learn Icon" className="w-6 h-6" />
            <span>Practise</span>
          </div>
          <FaChevronDown
            className={`transition-transform duration-300 ${
              practiseOpen ? "rotate-180" : ""
            }`}
          />
        </div>
        {practiseOpen && (
          <div className="pl-8 flex flex-col gap-1 text-xl">
            <Link to="/practise-Alphabet" className={`${isActive("/learn-Alphabet")} px-2 py-1 flex items-center gap-2`}>
              <img src={AlphabetBtn} alt="Alphabet Icon" className="w-6 h-6 bg-white" />
              Alphabet
            </Link>
            <Link to="/practise-Words" className={`${isActive("/learn-Words")} px-2 py-1 flex items-center gap-2`}>
              <img src={WordsBtn} alt="Words Icon" className="w-6 h-6 bg-white" />
              Words
            </Link>
          </div>
        )}

        {/* Achievements */}
        <Link
          to="/achievements"
          className={`px-4 py-2 text-2xl flex items-center gap-2 ${isActive("/achievements")}`}
        >
          <img src={AchievementsBtn} alt="Achievements Icon" className="w-6 h-6" />
          Achievements
        </Link>

        <hr className="border-gray-600 my-4" />

        {/* Help */}
        <Link
          to="/helpMenu"
          className={`px-4 py-2 text-2xl flex items-center gap-2 ${isActive(
            "/helpMenu"
          )}`}
        >
          <img src={HelpMenuBtn} alt="Help Icon" className="w-6 h-6" />
          Help
        </Link>

        {/* Settings */}
        <Link
          to="/settings"
          className={`px-4 py-2 text-2xl flex items-center gap-2 ${isActive(
            "/settings"
          )}`}
        >
          <img src={SettingsBtn} alt="Settings Icon" className="w-6 h-6" />
          Settings
        </Link>

        <hr className="border-gray-600 my-4" />

        {/* Logout button */}
        {isLoggedIn && (
          <button
            onClick={handleLogout}
            className="px-4 py-2 text-2xl text-white font-semibold flex items-center gap-2"
          >
            <img src={SignoutBtn} alt="Sign Out Icon" className="w-6 h-6" />
            Sign Out
          </button>
        )}
      </div>

      {/* Sign in / Register buttons (if logged out) */}
      {!isLoggedIn && (
        <div className="flex flex-col w-full gap-1">
          <Link
            to="/login"
            className="w-full px-4 py-2 text-2xl flex items-center gap-2 text-gray-200 hover:bg-[#1a436b] rounded"
          >
            <img src={SignInBtn} alt="SignIn Icon" className="w-6 h-6" />
            <span>Sign In</span>
          </Link>
          <Link
            to="/register"
            className="w-full px-4 py-2 text-2xl flex items-center gap-2 text-gray-200 hover:bg-[#1a436b] rounded mt-1"
          >
            <img src={RegisterBtn} alt="Register Icon" className="w-6 h-6" />
            <span>Register</span>
          </Link>
        </div>
      )}
    </div>
  );
};

export default SideNavbar;

