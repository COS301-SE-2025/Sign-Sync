import express from 'express';

const router = express.Router();

router.post('/getAnimation', async (req, res) => 
{
    const {word} = req.body;

    try 
    {
        const response = await req.app.locals.signCollection.findOne({ keywords : word });

        if(!response){
            let array = [];
            for (let i = 0; i < word.length; i++) {
                array.push(word[i].toUpperCase());
            }

            return res.status(200).json({
                status: 'success',
                response: array,
            });
        }else{
            return res.status(200).json({
                status: 'success',
                animation: response.animation,
            });
        }
    } 
    catch(error) 
    {
        return res.status(500).json({ message: 'Error finding translation', error: error.message });
    }
});

export default router;