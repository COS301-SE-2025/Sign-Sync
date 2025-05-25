import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import SideNavbar from './components/sideNavbar';
import LoginPage from './pages/LoginPage';
import RegistrationPage from './pages/RegistrationPage';

class app extends React.Component 
{
  render() 
  {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<SideNavbar />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegistrationPage />} />
        </Routes>
      </BrowserRouter>
    );
  }
}

export default app;
