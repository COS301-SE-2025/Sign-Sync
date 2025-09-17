import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { toast } from "react-toastify";
import { FaChevronDown } from "react-icons/fa";

import translateBtn from "../assets/Translator-icon.png";
import SettingsBtn from "../assets/Settings-icon.png";
import HelpMenuBtn from "../assets/info-icon.png";

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
    <div className="w-64 flex flex-col h-screen items-start px-0 pt-0 pb-5 bg-[#102a46]">
      <div className="w-full flex flex-col gap-2 pt-6 text-white">
        {/* App title */}
        <div className="px-4 py-2 text-3xl font-bold">Sign Sync</div>

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
          <span>Learn</span>
          <FaChevronDown
            className={`transition-transform duration-300 ${
              learnOpen ? "rotate-180" : ""
            }`}
          />
        </div>
        {learnOpen && (
          <div className="pl-8 flex flex-col gap-1 text-xl">
            <Link to="/learn-Alphabet" className={`${isActive("/learn-Alphabet")} px-2 py-1`}>
              Alphabet
            </Link>
            <Link to="/learn-Words" className={`${isActive("/learn-Words")} px-2 py-1`}>
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
          <span>Practise</span>
          <FaChevronDown
            className={`transition-transform duration-300 ${
              practiseOpen ? "rotate-180" : ""
            }`}
          />
        </div>
        {practiseOpen && (
          <div className="pl-8 flex flex-col gap-1 text-xl">
            <Link to="/practise-Alphabet" className={`${isActive("/practise-Alphabet")} px-2 py-1`}>
              Alphabet
            </Link>
            <Link to="/practise-Words" className={`${isActive("/practise-Words")} px-2 py-1`}>
              Words
            </Link>
          </div>
        )}

        {/* Achievements */}
        <Link
          to="/achievements"
          className={`px-4 py-2 text-2xl ${isActive("/achievements")}`}
        >
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

        {/* Logout button */}
        {isLoggedIn && (
          <button
            onClick={handleLogout}
            className="px-4 py-2 mt-2 text-2xl text-white font-semibold rounded text-left"
          >
            Sign Out
          </button>
        )}
      </div>

      {/* Sign in / Register buttons (if logged out) */}
      {!isLoggedIn && (
        <div className="flex w-full items-center justify-center gap-2 mt-auto px-2">
          <Link
            to="/login"
            className="text-xl flex items-center justify-center h-12 w-[120px] bg-white text-[#801d1f] font-semibold rounded"
          >
            Sign in
          </Link>
          <Link
            to="/register"
            className="text-xl flex items-center justify-center h-12 w-[120px] bg-[#801d1f] text-white font-semibold rounded"
          >
            Register
          </Link>
        </div>
      )}
    </div>
  );
};

export default SideNavbar;

