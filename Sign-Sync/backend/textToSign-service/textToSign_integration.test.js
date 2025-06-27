import request from 'supertest';
import express from 'express';
import textSignApi from './textSignApi';

// Set up a mock Express app
const app = express();
app.use(express.json());
app.use('/api', textSignApi);

// Integration Test 1: Successful response when word is found in the database
test('should return animation when word is found in the database', async () => {
  // Mock the database response
  const mockWord = 'hello';
  const mockAnimation = { animation: 'some animation data' };
  
  // Mock the behavior of the database collection
  app.locals.signCollection = {
    findOne: jest.fn().mockResolvedValueOnce(mockAnimation),
  };

  // Send request and check response
  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  expect(response.status).toBe(200);
  expect(response.body.status).toBe('success');
  expect(response.body.response).toBe(mockAnimation.animation);
});

// Integration Test 2: Successful response when word is not found in the database
test('should return uppercase word letters when word is not found in the database', async () => {
  // Mock database response for no data
  const mockWord = 'test';
  const mockArray = ['T', 'E', 'S', 'T'];
  
  app.locals.signCollection = {
    findOne: jest.fn().mockResolvedValueOnce(null),  // Simulate no data found
  };

  // Send request and check response
  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  expect(response.status).toBe(200);
  expect(response.body.status).toBe('success');
  expect(response.body.response).toEqual(mockArray);
});

// Integration Test 3: Error handling when the database query fails
test('should return 500 error when an exception occurs', async () => {
  const mockWord = 'error';
  
  // Simulate an error during the database query
  app.locals.signCollection = {
    findOne: jest.fn().mockRejectedValueOnce(new Error('Database error')),
  };

  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  expect(response.status).toBe(500);
  expect(response.body.message).toBe('Error finding translation');
  expect(response.body.error).toBe('Database error');
});
