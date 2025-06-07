import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegistrationPage from './pages/RegistrationPage';
import Translator from "./pages/TranslatorPage";
import SettingsPage from './pages/SettingsPage';

class app extends React.Component 
{
  render() 
  {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/translator" element={<Translator />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegistrationPage />} />
          <Route path="/settings" element={<SettingsPage />} />

        </Routes>
      </BrowserRouter>
    );
  }
}

export default app;
