const attempts = {}
const MAX_ATTEMPTS = 3
const BLOCK_TIME = 60 * 1000

function loginRateLimiter(req, res, next) {
    const ip = req.ip || req.connection.remoteAddress;
    const now = Date.now();

    if (!attempts[ip]) {
        attempts[ip] = { count: 0, lastAttempt: now, blockedUntil: 0 };
    }

    const entry = attempts[ip];

    if (now < entry.blockedUntil) {
        return res.status(429).json({ message: 'Too many login attempts. Please try again later.' });
    }

    if (now - entry.lastAttempt > BLOCK_TIME) {
        entry.count = 0;
    }

    entry.count++;
    entry.lastAttempt = now;

    if (entry.count > MAX_ATTEMPTS) {
        entry.blockedUntil = now + BLOCK_TIME;
        return res.status(429).json({ message: 'Too many login attempts. Please try again later.' });
    }

    next();
}

module.exports = loginRateLimiter;