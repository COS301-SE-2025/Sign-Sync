import express from 'express';
import bcrypt from 'bcrypt';

const router = express.Router();

router.post('/register', async (req, res) => 
{
    const { email, password } = req.body;

    try
    {
        const existingEmail = await req.app.locals.userCollection.findOne({ email })

        if(existingEmail)
        {
            return res.status(400).json({ message: 'Email already exists' });
        }

        const salt = 10;
        const hashedPassword = await bcrypt.hash(password, salt);

        const latestUser = await req.app.locals.userCollection
                            .find({})
                            .sort({userID: -1})
                            .limit(1)
                            .toArray();
        
        let newUserID;

        if(latestUser.length>0)
        {
            newUserID = latestUser[0].userID+1;
        }
        else
        {
            newUserID = 1;
        }

        const newUser = {
            userID: newUserID,
            email,
            password: hashedPassword,
        };

        await req.app.locals.userCollection.insertOne(newUser);

        return res.status(200).json({
            status: 'success',
            message: 'signup successful',
        });
    } 
    catch(error) 
    {
        res.status(500).json({ message: 'Error signing up user', error: error.message });
    }
    
}); 

router.post('/login', async (req, res) => 
{
    const { email, password } = req.body;

    try 
    {
        const user = await req.app.locals.userCollection.findOne({ email });

        if(!user) 
        {
            return res.status(400).json({ message: 'Email does not exist' });
        }

        const isMatch = await bcrypt.compare(password, user.password);

        if(!isMatch)
        {
            return res.status(401).json({ message: 'Incorrect password' });
        }

        const { password: _, ...userWithoutPassword } = user; //exclude password from response

        return res.status(200).json({
            status: 'success',
            message: 'Login successful',
            user: userWithoutPassword,
        });

    } 
    catch(error) 
    {
        return res.status(500).json({ message: 'Error logging in', error: error.message });
    }
});

router.get('/preferences/:userID', async (req, res) => 
{
    const { userID } = req.params;

    try 
    {
        const user = await req.app.locals.userCollection.findOne({ userID: parseInt(userID) });

        if(!user) 
        {
            return res.status(404).json({ message: 'User not found' });
        }

        //console.log("Fetched user for preferences:", user);

        res.status(200).json({
            status: 'success',
            preferences: user.preferences || {},
        });
    } 
    catch(error) 
    {
        res.status(500).json({ message: 'Error fetching preferences', error: error.message });
    }
});

router.put('/preferences/:userID', async (req, res) => 
{
    const { userID } = req.params;
    const updatedPreferences = req.body;

    try 
    {
        const result = await req.app.locals.userCollection.updateOne(
            { userID: parseInt(userID) },
            { $set: { preferences: updatedPreferences } }
        );

        if(result.matchedCount === 0) 
        {
            return res.status(404).json({ message: 'User not found' });
        }

        res.status(200).json({ status: 'success', message: 'Preferences updated' });
    } 
    catch(error) 
    {
        res.status(500).json({ message: 'Error updating preferences', error: error.message });
    }
});


export default router;