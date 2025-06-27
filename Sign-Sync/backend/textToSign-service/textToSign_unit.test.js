import request from 'supertest';
import express from 'express';
import textSignApi from './textSignApi';

// Set up a mock Express app
const app = express();
app.use(express.json()); // Parse JSON request bodies
app.use('/api', textSignApi);  // Use the API router

// Mock the database collection and its methods
const mockFindOne = jest.fn();
app.locals.signCollection = { findOne: mockFindOne };

// Unit Test 1: Successful response when word exists in the database
test('should return animation when word is found in the database', async () => {
  // Arrange
  const mockWord = 'hello';
  const mockAnimation = { animation: 'some animation data' };
  mockFindOne.mockResolvedValueOnce(mockAnimation); // Mock database response

  // Act
  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  // Assert
  expect(response.status).toBe(200);
  expect(response.body.status).toBe('success');
  expect(response.body.response).toBe(mockAnimation.animation);
  expect(mockFindOne).toHaveBeenCalledWith({ keywords: mockWord });
});

// Unit Test 2: Successful response when word is not found in the database
test('should return uppercase word letters when word is not found in the database', async () => {
  // Arrange
  const mockWord = 'test';
  const mockArray = ['T', 'E', 'S', 'T'];
  mockFindOne.mockResolvedValueOnce(null);  // Mock that no matching word was found

  // Act
  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  // Assert
  expect(response.status).toBe(200);
  expect(response.body.status).toBe('success');
  expect(response.body.response).toEqual(mockArray);
  expect(mockFindOne).toHaveBeenCalledWith({ keywords: mockWord });
});

// Unit Test 3: Error response when an exception occurs
test('should return 500 error when database query fails', async () => {
  // Arrange
  const mockWord = 'error';
  mockFindOne.mockRejectedValueOnce(new Error('Database error'));  // Simulate error

  // Act
  const response = await request(app)
    .post('/api/getAnimation')
    .send({ word: mockWord });

  // Assert
  expect(response.status).toBe(500);
  expect(response.body.message).toBe('Error finding translation');
  expect(response.body.error).toBe('Database error');
});
