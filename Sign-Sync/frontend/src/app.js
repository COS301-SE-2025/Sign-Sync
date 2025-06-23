import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import LoginPage from './pages/LoginPage';
import RegistrationPage from './pages/RegistrationPage';
import Translator from "./pages/TranslatorPage";
import SettingsPage from './pages/SettingsPage';
import HelpMenuPage from './pages/HelpMenuPage';
import ProductOverview from "./pages/articles/ProductOverview";
import FirstStepsTutorial from "./pages/articles/FirstStepsTutorial";
import SettingUpYourAccount from "./pages/articles/SettingUpYourAccount";
import AccountSettings from "./pages/articles/AccountSettings";
import BillingQuestions from "./pages/articles/BillingQuestions";
import Troubleshooting from "./pages/articles/Troubleshooting";

class app extends React.Component 
{
  render() 
  {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Translator />} />
          <Route path="/translator" element={<Translator />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegistrationPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/helpMenu" element={<HelpMenuPage />} />
          <Route path="/productOverview" element={<ProductOverview  />} />
          <Route path="/firstStepsTutorial" element={<FirstStepsTutorial  />} />
          <Route path="/settingUpYourAccount" element={<SettingUpYourAccount  />} />
          <Route path="/accountSettings" element={<AccountSettings  />} />
          <Route path="/billingQuestions" element={<BillingQuestions  />} />
          <Route path="/troubleshooting" element={<Troubleshooting  />} />
        </Routes>
      </BrowserRouter>
    );
  }
}

export default app;
