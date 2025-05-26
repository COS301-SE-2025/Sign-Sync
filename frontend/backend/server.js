import express from 'express';
import path from 'path';

const app = express();
const port = process.env.PORT || 3000;

// Serve static files from the frontend/public directory
const publicPath = path.resolve('frontend', 'public');
app.use(express.static(publicPath));

// For any other route, serve index.html (for React Router)
app.get('*', (req, res) => {
  res.sendFile(path.resolve(publicPath, 'index.html'));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});