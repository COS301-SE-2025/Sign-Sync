import express from 'express';
import path from 'path';
import { MongoClient } from 'mongodb';
import apiRoutes from './userApi'; // keep as-is (relative to dist)
import textSignApiRoutes from './textSignApi';
import dotenv from 'dotenv';

dotenv.config();

const app = express();

const url = process.env.MONGODB_URI || 'mongodb://localhost:27017';
const dbName = process.env.MONGODB_DB_NAME || 'testdb';
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Serve frontend files
const publicPath = path.resolve('frontend', 'public');
app.use(express.static(publicPath));

// API routes
app.use('/userApi', apiRoutes);
app.use('/TextSignApi', textSignApiRoutes);

// Fallback for React Router
app.get('*', (req, res) => {
  res.sendFile(path.resolve(publicPath, 'index.html'));
});

// Start server immediately
app.listen(port, () => {
  console.log(`✅ Server running at http://localhost:${port}`);
});

// Connect to MongoDB asynchronously (won’t block server start)
(async () => {
  try {
    const client = new MongoClient(url);
    await client.connect();
    console.log('✅ Connected to MongoDB');

    const db = client.db(dbName);
    app.locals.userCollection = db.collection('Users');
    app.locals.signCollection = db.collection('Signs');
  } catch (err) {
    console.error('⚠️ MongoDB connection failed (server still running):', err.message);
  }
})();




