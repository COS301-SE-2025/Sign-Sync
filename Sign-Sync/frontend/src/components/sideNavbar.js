import React from "react";
import { Link } from "react-router-dom";

import topBtn from "../assets/sideNav-topButton.png";
import translateBtn from '../assets/Translator-icon.png';
import EducationBtn from '../assets/Education-icon.png';
import SettingsBtn from '../assets/Settings-icon.png';
import HelpMenuBtn from '../assets/info-icon.png';

class SideNavbar extends React.Component 
{
  constructor(props)
  {
    super(props);
    this.state = {
      isLoggedIn: false,
    };
  }

  componentDidMount()
  {
    const user = localStorage.getItem('user');
    
    if(user) 
    {
      this.setState({ isLoggedIn: true });
    }
  } 

  handleLogout = () =>
  {
    localStorage.removeItem('user');
    this.setState({ isLoggedIn: false });

    alert("Logout successful!, redirecting to Splash page...");

    window.location.href = '/';
  }

  render()  
  {
    const { isLoggedIn } = this.state;

    return (
      <div className="w-64 flex flex-col h-screen items-start px-0 pt-0 pb-5 bg-[#102a46]"> 
        <div className="relative w-full h-[66px] bg-[#1c4a7c]">
          <div className="flex w-[61px] h-[60px] items-center justify-center relative top-[3px] left-[13px]">
            <div className="flex flex-col w-24 items-center justify-center relative mt-[-18.00px] mb-[-18.00px] ml-[-17.50px] mr-[-17.50px] rounded-[28px] overflow-hidden">
              <div className="flex h-24 items-center justify-center relative self-stretch w-full">
                <div className="w-12 h-12" />
                  <img src={topBtn} alt="Logo" className="w-15 h-13 pt-2" />
              </div>
            </div>
          </div>
        </div>

        <div className="w-full flex flex-col gap-4 pt-4 text-white ">
            <div className="p-2 text-4xl">
                <Link to="/translator">
                  <img src={translateBtn} alt="Translator Icon" className="w-8 h-8 inline-block mr-4" />
                  Translator
                </Link>
            </div>
            <div className="p-2 text-4xl">
              <img src={EducationBtn} alt="Translator Icon" className="w-8 h-8 inline-block mr-4" />
              Education
            </div>
            <div className="p-2 text-4xl">
              <Link to="/settings">
                <img src={SettingsBtn} alt="Translator Icon" className="w-8 h-8 inline-block mr-4" />
                Settings
              </Link>
            </div>
            <div className="p-2 text-4xl">
              <Link to="/helpMenu">
                <img src={HelpMenuBtn} alt="Translator Icon" className="w-8 h-8 inline-block mr-4" />
                Help Menu
              </Link>
            </div>
        </div>

        <div className="flex w-full h-[78px] items-center justify-center gap-4 mt-auto px-2">

          {!isLoggedIn ? (
            <>
              <Link to="/login" className=" text-2xl flex items-center justify-center h-12 w-[129px] bg-white text-[#801d1f] font-semibold rounded">
                Sign in
              </Link>
              <Link to="/register" className=" text-2xl flex items-center justify-center h-12 w-[129px] bg-[#801d1f] text-white font-semibold rounded">
                Register
              </Link>
            </>
          ) : (
            <button 
              onClick={this.handleLogout} 
              className="text-2xl h-12 w-[200px] bg-red-700 text-white font-semibold rounded hover:bg-red-800"
            >
              Sign out
            </button>
          )}
        </div>
      </div>
    );
  }
}

export default SideNavbar;
