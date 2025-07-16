import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import LandingPage from './pages/LandingPage';
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
import EmailSupport from "./pages/articles/EmailSupport";
import LiveChatSupport from "./pages/articles/LiveChatSupport";
import PhoneSupport from "./pages/articles/PhoneSupport";
import AdvancedTechniques from "./pages/articles/AdvancedTechniques";
import BasicFeatures from "./pages/articles/BasicFeatures";
import ContactSupport from "./pages/articles/ContactSupport";
import DashboardGuide from "./pages/articles/DashboardGuide";
import DataExport from "./pages/articles/DataExport";
import Integrations from "./pages/articles/Integrations";
import PasswordReset from "./pages/articles/PasswordReset";
import PrivacySettings from "./pages/articles/PrivacySettings";
import ProductivityTips from "./pages/articles/ProductivityTips";
import LearnAlphabet from "./pages/LearnAlphabetPage";
import PractiseAlphabet from "./pages/PractiseAlphabetPage";

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
          <Route path="/learn-Alphabet" element={<LearnAlphabet/>} />
          <Route path="/practise-Alphabet" element={<PractiseAlphabet />} />
          <Route path="/helpMenu" element={<HelpMenuPage />} />
          <Route path="/productOverview" element={<ProductOverview  />} />
          <Route path="/firstStepsTutorial" element={<FirstStepsTutorial  />} />
          <Route path="/settingUpYourAccount" element={<SettingUpYourAccount  />} />
          <Route path="/accountSettings" element={<AccountSettings  />} />
          <Route path="/billingQuestions" element={<BillingQuestions  />} />
          <Route path="/troubleshooting" element={<Troubleshooting  />} />
          <Route path="/emailSupport" element={<EmailSupport  />} />
          <Route path="/liveChatSupport" element={<LiveChatSupport  />} />
          <Route path="/phoneSupport" element={<PhoneSupport  />} />
          <Route path="/advancedTechniques" element={<AdvancedTechniques  />} />
          <Route path="/basicFeatures" element={<BasicFeatures  />} />
          <Route path="/contactSupport" element={<ContactSupport  />} />
          <Route path="/dashboardGuide" element={<DashboardGuide  />} />
          <Route path="/dataExport" element={<DataExport  />} />
          <Route path="/integrations" element={<Integrations  />} />
          <Route path="/passwordReset" element={<PasswordReset  />} />
          <Route path="/privacySettings" element={<PrivacySettings  />} />
          <Route path="/productivityTips" element={<ProductivityTips  />} />
        </Routes>
      </BrowserRouter>
    );
  }
}

export default app;
