import express from 'express';

const router = express.Router();

router.post('/register', async (req, res) => 
{
    const { username, email, password } = req.body;

    try
    {
        const existingUsername = await req.app.locals.userCollection.findOne({ username })
        const existingEmail = await req.app.locals.userCollection.findOne({ email })

        if(existingUsername) 
        {
            return res.status(400).json({ message: 'Username already exists' });
        }

        if(existingEmail)
        {
            return res.status(400).json({ message: 'Email already exists' });
        }

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
            username,
            email,
            password
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
    const { username, email, password } = req.body;

    try 
    {
        const user = await req.app.locals.userCollection.findOne({ username, email});

        if(!user) 
        {
            return res.status(400).json({ message: 'Invalid username or email' });
        }

        if(user.password !== password) 
        {
            return res.status(401).json({ message: 'Incorrect password' });
        }

        return res.status(200).json({
            status: 'success',
            message: 'Login successful',
            user,
        });

    } 
    catch(error) 
    {
        return res.status(500).json({ message: 'Error logging in', error: error.message });
    }
});


export default router;