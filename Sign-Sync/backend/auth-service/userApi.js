import express from 'express';
import bcrypt from 'bcrypt';

const router = express.Router();

const loginRateLimiter = require('../auth-service/loginRateLimiter.js');

router.post('/register', async (req, res) => {
    const { email, password } = req.body;

    try {
        const existingEmail = await req.app.locals.userCollection.findOne({ email })

        const defaultPreference = {
            displayMode: 'Light Mode',
            preferredAvatar: 'Zac',
            animationSpeed: 'Normal',
            fontSize: 'Medium',
            speechSpeed: 1,
            speechVoice: 'George'
        };

        if (existingEmail) {
            return res.status(400).json({ message: 'Email already exists' });
        }

        const salt = 10;
        const hashedPassword = await bcrypt.hash(password, salt);

        const latestUser = await req.app.locals.userCollection
            .find({})
            .sort({ userID: -1 })
            .limit(1)
            .toArray();

        let newUserID;

        if (latestUser.length > 0) {
            newUserID = latestUser[0].userID + 1;
        }
        else {
            newUserID = 1;
        }

        const newUser = {
            userID: newUserID,
            email,
            password: hashedPassword,
            achievements: [1],
            preferences: defaultPreference,
        };

        await req.app.locals.userCollection.insertOne(newUser);

        return res.status(200).json({
            status: 'success',
            message: 'signup successful',
        });
    }
    catch (error) {
        res.status(500).json({ message: 'Error signing up user', error: error.message });
    }

});

router.post('/login', loginRateLimiter, async (req, res) => {
    const { email, password } = req.body;

    try {
        const user = await req.app.locals.userCollection.findOne({ email });

        if (!user) {
            return res.status(400).json({ message: 'Email does not exist' });
        }

        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.status(401).json({ message: 'Incorrect password' });
        }

        const { password: _, ...userWithoutPassword } = user; //exclude password from response

        return res.status(200).json({
            status: 'success',
            message: 'Login successful',
            user: userWithoutPassword,
        });

    }
    catch (error) {
        return res.status(500).json({ message: 'Error logging in', error: error.message });
    }
});

router.delete('/deleteAccount/:userID', async (req, res) => {
    const { userID } = req.params;

    try {
        const result = await req.app.locals.userCollection.deleteOne({ userID: parseInt(userID) });

        if (result.deletedCount === 0) {
            return res.status(404).json({ message: 'User not found or already deleted' });
        }

        res.status(200).json({ status: 'success', message: 'User account deleted successfully' });
    }
    catch (error) {
        res.status(500).json({ message: 'Error deleting user', error: error.message });
    }
});

router.get('/preferences/:userID', async (req, res) => {
    const { userID } = req.params;

    try {
        const user = await req.app.locals.userCollection.findOne({ userID: parseInt(userID) });

        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        //console.log("Fetched user for preferences:", user);

        res.status(200).json({
            status: 'success',
            preferences: user.preferences || {},
        });
    }
    catch (error) {
        res.status(500).json({ message: 'Error fetching preferences', error: error.message });
    }
});

router.put('/preferences/:userID', async (req, res) => {
    const { userID } = req.params;
    const updatedPreferences = req.body;

    try {
        const result = await req.app.locals.userCollection.updateOne(
            { userID: parseInt(userID) },
            { $set: { preferences: updatedPreferences } }
        );

        if (result.matchedCount === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        res.status(200).json({ status: 'success', message: 'Preferences updated' });
    }
    catch (error) {
        res.status(500).json({ message: 'Error updating preferences', error: error.message });
    }
});

/**
 * @swagger
 * tags:
 *   name: Achievements
 *   description: User achievements management
 */

/**
 * @swagger
 * /achievements/{userID}:
 *   get:
 *     summary: Get user achievements
 *     description: Retrieve all achievements for a specific user
 *     tags: [Achievements]
 *     parameters:
 *       - in: path
 *         name: userID
 *         schema:
 *           type: string
 *         required: true
 *         description: The ID of the user whose achievements to retrieve
 *     responses:
 *       200:
 *         description: A list of user achievements
 *         content:
 *           application/json:
 *             schema:
 *               type: array
 *               items:
 *                 $ref: '#/components/schemas/Achievement'
 *       404:
 *         description: User not found
 *       500:
 *         description: Server error
 */
router.get('/achievements/:userID', async (req, res) => {
    const { userID } = req.params;

    try {
        const user = await req.app.locals.userCollection.findOne({
            userID: parseInt(userID)
        });

        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        // Set Content-Type before sending response
        res.set('Content-Type', 'application/json');
        res.status(200).json(user.achievements || []);
    } catch (error) {
        res.status(500).json({
            message: 'Error getting achievements',
            error: error.message
        });
    }
});

/**
 * @swagger
 * /achievements/{userID}:
 *   put:
 *     summary: Update user achievements
 *     description: Update or add achievements for a specific user
 *     tags: [Achievements]
 *     parameters:
 *       - in: path
 *         name: userID
 *         schema:
 *           type: string
 *         required: true
 *         description: The ID of the user whose achievements to update
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/AchievementUpdate'
 *     responses:
 *       200:
 *         description: Achievements updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Achievement'
 *       400:
 *         description: Invalid input
 *       404:
 *         description: User not found
 *       500:
 *         description: Server error
 */
router.put('/achievements/:userID', async (req, res) => {
    const { userID } = req.params;
    const { newAchievements } = req.body;

    try {
        // Update in database
        const result = await req.app.locals.userCollection.updateOne(
            { userID: parseInt(userID) },
            { $set: { achievements: newAchievements } }  // Correct field name
        );

        if (result.matchedCount === 0) {
            return res.status(404).json({ message: 'User not found' });
        }

        // Return success response
        res.set('Content-Type', 'application/json');
        res.status(200).json({
            status: 'success',
            message: 'Achievements updated successfully',
            achievements
        });
    } catch (error) {
        res.status(500).json({
            message: 'Error updating achievements',
            error: error.message
        });
    }
});


export default router;