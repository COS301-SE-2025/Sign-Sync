import request from 'supertest';
import express from 'express';
import { MongoClient } from 'mongodb';
import router from '../backend/userApi.js'; // Update with your actual file path

describe('User Authentication Routes', () => {
  let app;
  let connection;
  let db;
  let userCollection;

  beforeAll(async () => {
    // Setup test MongoDB
    connection = await MongoClient.connect(global.__MONGO_URI__, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    db = await connection.db(global.__MONGO_DB_NAME__);
    userCollection = db.collection('users');

    // Create Express app
    app = express();
    app.use(express.json());
    app.locals.userCollection = userCollection;
    app.use('/auth', router);
  });

  afterAll(async () => {
    await connection.close();
  });

  beforeEach(async () => {
    // Clear the users collection before each test
    await userCollection.deleteMany({});
  });

  describe('POST /auth/register', () => {
    it('should register a new user successfully', async () => {
      const newUser = {
        username: 'testuser',
        email: 'test@example.com',
        password: 'password123',
      };

      const response = await request(app)
        .post('/auth/register')
        .send(newUser);

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('signup successful');

      // Verify user was actually created in DB
      const user = await userCollection.findOne({ username: 'testuser' });
      expect(user).toBeTruthy();
      expect(user.email).toBe('test@example.com');
      expect(user.userID).toBe(1); // First user should have ID 1
    });

    it('should return 400 if username already exists', async () => {
      // Insert a test user first
      await userCollection.insertOne({
        userID: 1,
        username: 'existinguser',
        email: 'existing@example.com',
        password: 'password123',
      });

      const response = await request(app)
        .post('/auth/register')
        .send({
          username: 'existinguser',
          email: 'new@example.com',
          password: 'password123',
        });

      expect(response.status).toBe(400);
      expect(response.body.message).toBe('Username already exists');
    });

    it('should return 400 if email already exists', async () => {
      // Insert a test user first
      await userCollection.insertOne({
        userID: 1,
        username: 'user1',
        email: 'existing@example.com',
        password: 'password123',
      });

      const response = await request(app)
        .post('/auth/register')
        .send({
          username: 'newuser',
          email: 'existing@example.com',
          password: 'password123',
        });

      expect(response.status).toBe(400);
      expect(response.body.message).toBe('Email already exists');
    });

    it('should auto-increment userID correctly', async () => {
      // Insert first user
      await request(app)
        .post('/auth/register')
        .send({
          username: 'user1',
          email: 'user1@example.com',
          password: 'password123',
        });

      // Insert second user
      const response = await request(app)
        .post('/auth/register')
        .send({
          username: 'user2',
          email: 'user2@example.com',
          password: 'password123',
        });

      // Verify second user has ID 2
      const user = await userCollection.findOne({ username: 'user2' });
      expect(user.userID).toBe(2);
    });

    it('should return 500 if there is a database error', async () => {
      // Simulate a database error by closing the connection
      await connection.close();

      const response = await request(app)
        .post('/auth/register')
        .send({
          username: 'testuser',
          email: 'test@example.com',
          password: 'password123',
        });

      expect(response.status).toBe(500);
      expect(response.body.message).toContain('Error signing up user');

      // Reconnect for other tests
      connection = await MongoClient.connect(global.__MONGO_URI__);
      db = await connection.db(global.__MONGO_DB_NAME__);
      userCollection = db.collection('users');
      app.locals.userCollection = userCollection;
    });
  });

  describe('POST /auth/login', () => {
    beforeEach(async () => {
      // Insert a test user for login tests
      await userCollection.insertOne({
        userID: 1,
        username: 'testuser',
        email: 'test@example.com',
        password: 'correctpassword',
      });
    });

    it('should login successfully with correct credentials (username)', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'testuser',
          password: 'correctpassword',
        });

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('Login successful');
      expect(response.body.user).toBeTruthy();
      expect(response.body.user.username).toBe('testuser');
    });

    it('should login successfully with correct credentials (email)', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'correctpassword',
        });

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('Login successful');
      expect(response.body.user).toBeTruthy();
      expect(response.body.user.email).toBe('test@example.com');
    });

    it('should return 400 if username/email is invalid', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'nonexistent',
          password: 'correctpassword',
        });

      expect(response.status).toBe(400);
      expect(response.body.message).toBe('Invalid username or email');
    });

    it('should return 401 if password is incorrect', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'testuser',
          password: 'wrongpassword',
        });

      expect(response.status).toBe(401);
      expect(response.body.message).toBe('Incorrect password');
    });

    it('should return 500 if there is a database error', async () => {
      // Simulate a database error by closing the connection
      await connection.close();

      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'testuser',
          password: 'correctpassword',
        });

      expect(response.status).toBe(500);
      expect(response.body.message).toContain('Error logging in');

      // Reconnect for other tests
      connection = await MongoClient.connect(global.__MONGO_URI__);
      db = await connection.db(global.__MONGO_DB_NAME__);
      userCollection = db.collection('users');
      app.locals.userCollection = userCollection;
    });
  });
});