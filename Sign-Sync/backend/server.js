import express from 'express';
import path from 'path';
import { MongoClient } from 'mongodb';
import apiRoutes from './userApi'; //this seems incorrect, but it is correct. It imports the userApi.js file relative to the dist. Leave it as is.
import textSignApiRoutes from './textSignApi'; // The text to sign api

import dotenv from 'dotenv';

dotenv.config(); //load environment variables from .env file

const app = express();

const url = process.env.MONGODB_URI;
const dbName = process.env.MONGODB_DB_NAME;
const port = process.env.PORT;

//Middleware
app.use(express.json());

//Serve frontend files
const publicPath = path.resolve('frontend', 'public');
app.use(express.static(publicPath));

app.use('/userApi', apiRoutes);
app.use('/TextSignApi',textSignApiRoutes);

//Fallback for React Router
app.get('*', (req, res) => {
  res.sendFile(path.resolve(publicPath, 'index.html'));
});

//MongoDB connection and server start
async function main() 
{
  const client = new MongoClient(url);

  try 
  {
    await client.connect();

    const db = client.db(dbName);

    const userCollection = db.collection('Users');
    const signCollection = db.collection('Signs');

    app.locals.userCollection = userCollection;
    app.locals.signCollection = signCollection;

    app.listen(port, () => {
      console.log(`Server running at http://localhost:${port}`);
    });

  } 
  catch(err) 
  {
    console.error('MongoDB connection failed:', err);
  }
}

main().catch(console.error);

