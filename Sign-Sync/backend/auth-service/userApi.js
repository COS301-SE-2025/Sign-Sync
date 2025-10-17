import express from 'express';
import bcrypt from 'bcrypt';

const router = express.Router();

const loginRateLimiter = require('./loginRateLimiter.js');

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

        const defaultAchievements = {
            firstLetter: false,
            firstWord: false,
            allWords: false,
            allLetters: false,
            firstLogin: false,
            bronze: false,
            silver: false,
            gold: false,
            platinum: false
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
            achievements: defaultAchievements,
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

        await req.app.locals.userCollection.updateOne(
            { email },
            { $set: { 'achievements.firstLogin': true } }
        );

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


// put this near the top, replace BLANK and helpers
const DEFAULTS = {
    firstLogin: false,
    firstLetter: false,
    firstWord: false,
    bronze: false,
    allLetters: false,
    silver: false,
    allWords: false,
    gold: false,
    platinum: false,
};

function booleansToIds(b = {}) {
    const map = {
        firstLogin: 1,
        firstLetter: 2,
        firstWord: 3,
        bronze: 4,
        allLetters: 5,
        silver: 6,
        allWords: 7,
        gold: 8,
        platinum: 9,
    };
    return Object.keys(map).filter(k => !!b[k]).map(k => map[k]);
}

function idsToBooleans(ids = []) {
    return {
        firstLogin: ids.includes(1),
        firstLetter: ids.includes(2),
        firstWord: ids.includes(3),
        bronze: ids.includes(4),
        allLetters: ids.includes(5),
        silver: ids.includes(6),
        allWords: ids.includes(7),
        gold: ids.includes(8),
        platinum: ids.includes(9),
    };
}

// Accept wrapped/unwrapped shapes and always return a full boolean object
function normalizeBooleans(anyShape = {}) {
    const data = (anyShape && typeof anyShape === 'object' && 'achievements' in anyShape)
        ? anyShape.achievements
        : anyShape;

    if (Array.isArray(data)) return { ...DEFAULTS, ...idsToBooleans(data) };
    return { ...DEFAULTS, ...(data || {}) };
}



// const BLANK = { firstLetter:false, firstWord:false, allLetters:false, allWords:false };

// function booleansToIds(b = {}) {
//   const map = {
//     firstLogin: 1,
//     firstLetter: 2,
//     firstWord: 3,
//     bronze: 4,
//     allLetters: 5,
//     silver: 6,
//     allWords: 7,
//     gold: 8,
//     platinum: 9,
//   };
//   return Object.keys(map).filter(k => !!b[k]).map(k => map[k]);
// }

// function idsToBooleans(ids = []) {
//   return {
//     firstLogin: ids.includes(1),
//     firstLetter: ids.includes(2),
//     firstWord: ids.includes(3),
//     bronze: ids.includes(4),
//     allLetters: ids.includes(5),
//     silver: ids.includes(6),
//     allWords: ids.includes(7),
//     gold: ids.includes(8),
//     platinum: ids.includes(9),
//   };
// }

// function normalizeBooleans(b = {}) {
//   return { ...BLANK, ...b };
// }

// ---------------- GET /achievements/:userID ----------------
router.get('/achievements/:userID', async (req, res) => {
    const userID = parseInt(req.params.userID, 10);
    try {
        const user = await req.app.locals.userCollection.findOne(
            { userID },
            { projection: { achievements: 1, _id: 0 } }
        );
        if (!user) return res.status(404).json({ message: 'User not found' });

        const booleans = normalizeBooleans(user.achievements);
        res.set('Content-Type', 'application/json');
        return res.status(200).json(booleans);
    } catch (error) {
        return res.status(500).json({ message: 'Error getting achievements', error: error.message });
    }
});


// ---------------- PUT /achievements/:userID ----------------
router.put('/achievements/:userID', async (req, res) => {
    const userID = parseInt(req.params.userID, 10);
    try {
        // Load current so we can merge (prevents wiping missing keys)
        const currentUser = await req.app.locals.userCollection.findOne(
            { userID },
            { projection: { achievements: 1, _id: 0 } }
        );
        if (!currentUser) return res.status(404).json({ message: 'User not found' });

        // Accept either { newAchievements:[ids] } OR a boolean object (optionally wrapped)
        const incoming = Array.isArray(req.body?.newAchievements)
            ? idsToBooleans(req.body.newAchievements)
            : normalizeBooleans(req.body);

        const merged = { ...DEFAULTS, ...(currentUser.achievements || {}), ...incoming };

        await req.app.locals.userCollection.updateOne(
            { userID },
            { $set: { achievements: merged } }
        );

        res.set('Content-Type', 'application/json');
        return res.status(200).json(merged); // return booleans as source of truth
    } catch (error) {
        return res.status(500).json({ message: 'Error updating achievements', error: error.message });
    }
});




export default router;