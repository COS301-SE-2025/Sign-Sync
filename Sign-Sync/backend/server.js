import express from 'express';
import path from 'path';
import { MongoClient } from 'mongodb';

const app = express();
const port = process.env.PORT || 3000;

// MongoDB connection string
const url = 'mongodb+srv://Jamean:g19Apo11oProjects@sign-sync.wxahktp.mongodb.net/'; //this is unsafe, but for development purposes only
const dbName = 'Sign-Sync';

// Middleware
app.use(express.json());

// Serve frontend files
const publicPath = path.resolve('frontend', 'public');
app.use(express.static(publicPath));

// Fallback for React Router
app.get('*', (req, res) => {
  res.sendFile(path.resolve(publicPath, 'index.html'));
});

// MongoDB connection and server start
async function main() 
{
  const client = new MongoClient(url);

  try 
  {
    //console.log('Connecting to MongoDB...');
    await client.connect();
    //console.log('Connected to MongoDB!');

    const db = client.db(dbName);

    //console.log('Available collections:', await db.listCollections().toArray()); //debugging

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
