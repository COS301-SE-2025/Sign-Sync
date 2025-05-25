// import React from 'react';
// import ReactDOM from 'react-dom';
// import App from './app';

// const root = ReactDOM.createRoot(document.getElementById('root'));

// root.render(<App />);


import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './app';
import SideNavbar from './components/sideNavbar';

import './index.css';

const domNode = document.getElementById('root');
const root = createRoot(domNode);

// root.render(<App />);
root.render(<SideNavbar />);
