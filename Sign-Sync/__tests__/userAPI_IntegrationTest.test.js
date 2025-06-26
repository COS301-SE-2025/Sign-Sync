import request from 'supertest';
import express from 'express';
import { MongoClient } from 'mongodb';
import router from '../backend/dist/userApi.js'; 
import bcrypt from 'bcrypt';

describe('User API Routes', () => {
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
      const user = await userCollection.findOne({ email: 'test@example.com' });
      expect(user).toBeTruthy();
      expect(user.email).toBe('test@example.com');
      expect(user.userID).toBe(1);
      
      // Verify password is hashed
      const isMatch = await bcrypt.compare('password123', user.password);
      expect(isMatch).toBe(true);
    });

    it('should return 400 if email already exists', async () => {
      await userCollection.insertOne({
        userID: 1,
        email: 'existing@example.com',
        password: 'hashedpassword',
      });

      const response = await request(app)
        .post('/auth/register')
        .send({
          email: 'existing@example.com',
          password: 'password123',
        });

      expect(response.status).toBe(400);
      expect(response.body.message).toBe('Email already exists');
    });

    it('should auto-increment userID correctly', async () => {
      // First user
      await request(app)
        .post('/auth/register')
        .send({
          email: 'user1@example.com',
          password: 'password123',
        });

      // Second user
      const response = await request(app)
        .post('/auth/register')
        .send({
          email: 'user2@example.com',
          password: 'password123',
        });

      const user = await userCollection.findOne({ email: 'user2@example.com' });
      expect(user.userID).toBe(2);
    });
  });

  describe('POST /auth/login', () => {
    beforeEach(async () => {
      const hashedPassword = await bcrypt.hash('correctpassword', 10);
      await userCollection.insertOne({
        userID: 1,
        email: 'test@example.com',
        password: hashedPassword,
      });
    });

    it('should login successfully with correct credentials', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'correctpassword',
        });

      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('Login successful');
      expect(response.body.user.email).toBe('test@example.com');
      expect(response.body.user.password).toBeUndefined(); // Ensure password is excluded
    });

    it('should return 400 if email does not exist', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'correctpassword',
        });

      expect(response.status).toBe(400);
      expect(response.body.message).toBe('Email does not exist');
    });

    it('should return 401 if password is incorrect', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'wrongpassword',
        });

      expect(response.status).toBe(401);
      expect(response.body.message).toBe('Incorrect password');
    });

    it('should return 400 for missing email', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          password: 'correctpassword',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('DELETE /auth/deleteAccount/:userID', () => {
    beforeEach(async () => {
      await userCollection.insertOne({
        userID: 1,
        email: 'test@example.com',
        password: 'hashedpassword',
        preferences: { theme: 'dark' }
      });
    });

    it('should delete an account successfully', async () => {
      const response = await request(app)
        .delete('/auth/deleteAccount/1');
      
      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('User account deleted successfully');
      
      const user = await userCollection.findOne({ userID: 1 });
      expect(user).toBeNull();
    });

    it('should return 404 for non-existent user', async () => {
      const response = await request(app)
        .delete('/auth/deleteAccount/999');
      
      expect(response.status).toBe(404);
      expect(response.body.message).toBe('User not found or already deleted');
    });

    it('should handle invalid userID format', async () => {
      const response = await request(app)
        .delete('/auth/deleteAccount/invalid');
      
      expect(response.status).toBe(404);
    });
  });

  describe('GET /auth/preferences/:userID', () => {
    beforeEach(async () => {
      await userCollection.insertOne({
        userID: 1,
        email: 'test@example.com',
        password: 'hashedpassword',
        preferences: { theme: 'dark', notifications: true }
      });
    });

    it('should get user preferences successfully', async () => {
      const response = await request(app)
        .get('/auth/preferences/1');
      
      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.preferences).toEqual({
        theme: 'dark',
        notifications: true
      });
    });

    it('should return empty preferences if none set', async () => {
      await userCollection.updateOne(
        { userID: 1 },
        { $unset: { preferences: "" } }
      );
      
      const response = await request(app)
        .get('/auth/preferences/1');
      
      expect(response.status).toBe(200);
      expect(response.body.preferences).toEqual({});
    });

    it('should return 404 for non-existent user', async () => {
      const response = await request(app)
        .get('/auth/preferences/999');
      
      expect(response.status).toBe(404);
      expect(response.body.message).toBe('User not found');
    });
  });

  describe('PUT /auth/preferences/:userID', () => {
    beforeEach(async () => {
      await userCollection.insertOne({
        userID: 1,
        email: 'test@example.com',
        password: 'hashedpassword',
      });
    });

    it('should update preferences successfully', async () => {
      const newPreferences = { theme: 'light', notifications: false };
      
      const response = await request(app)
        .put('/auth/preferences/1')
        .send(newPreferences);
      
      expect(response.status).toBe(200);
      expect(response.body.status).toBe('success');
      expect(response.body.message).toBe('Preferences updated');
      
      const user = await userCollection.findOne({ userID: 1 });
      expect(user.preferences).toEqual(newPreferences);
    });

    it('should return 404 for non-existent user', async () => {
      const response = await request(app)
        .put('/auth/preferences/999')
        .send({ theme: 'light' });
      
      expect(response.status).toBe(404);
      expect(response.body.message).toBe('User not found');
    });

    it('should handle empty preferences object', async () => {
      const response = await request(app)
        .put('/auth/preferences/1')
        .send({});
      
      expect(response.status).toBe(200);
      
      const user = await userCollection.findOne({ userID: 1 });
      expect(user.preferences).toEqual({});
    });
  });

  describe('Database Error Handling', () => {
    it('should return 500 if there is a database error during registration', async () => {
      await connection.close();

      const response = await request(app)
        .post('/auth/register')
        .send({
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
});