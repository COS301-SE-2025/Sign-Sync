import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './app';

import './index.css';
import PreferenceManager from './components/PreferenceManager';
const domNode = document.getElementById('root');
const root = createRoot(domNode);

(async () => {
    await PreferenceManager.initialize();

    root.render(<App />);
})();


//root.render(<App />);
