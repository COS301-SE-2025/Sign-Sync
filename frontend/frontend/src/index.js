//u23524121 Jamean Groenewald

// import React from 'react';
// import ReactDOM from 'react-dom';
// import App from './app';

// const root = ReactDOM.createRoot(document.getElementById('root'));

// root.render(<App />);


import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './app';

const domNode = document.getElementById('root');
const root = createRoot(domNode);

root.render(<App />);
