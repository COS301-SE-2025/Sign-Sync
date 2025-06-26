const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

import { MongoMemoryServer } from 'mongodb-memory-server';

let mongoServer;

beforeAll(async () => {
  // Create in-memory MongoDB instance
  mongoServer = await MongoMemoryServer.create();
  const mongoUri = mongoServer.getUri();
  
  // Set global variables for tests to access
  process.env.MONGO_URI = mongoUri;
  global.__MONGO_URI__ = mongoUri;
  global.__MONGO_DB_NAME__ = 'testdb';
});

afterAll(async () => {
  // Stop the in-memory MongoDB
  await mongoServer.stop();
});