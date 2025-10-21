# TempleVision — Monument Structure Detection

TempleVision is a small research-focused web app that helps researchers, archaeologists and enthusiasts analyze monument and temple images/videos using a hybrid of classical vision and ML (YOLO). Upload an image or video, and the project will run object extraction, feature matching, and segmentation to reveal architectural structures and produce annotated outputs.

Key features
- Image upload & detection (Flask backend)
- Video upload & frame analysis (Flask backend + TempleStructureAnalyzer)
- Web dashboard and authentication (Express + MongoDB)
- Simple login/signup with JWT tokens
- Local UI for uploading images/videos and viewing results

Quick links
- Frontend: `frontend/login.html` and `frontend/index.html`
- Flask detection server: `backend.py` (runs on port 5000 by default)
- Express/MongoDB auth server: `frontend/index.js` (runs on port 4000)

Getting started (fast)
1. Clone this repo and open the project root:

   ```bash
   git clone <repo-url>
   cd softwareengg
   ```

2. Install Python deps (for detection backend):

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask detection backend (port 5000):

   ```bash
   python backend.py
   ```

4. Configure and start the Express + MongoDB server (frontend):

   - Copy `.env.example` (or edit `frontend/.env`) with your MongoDB and JWT secrets.
   - From project root run:

   ```bash
   npm install
   npm start
   ```

   This runs `cd frontend && nodemon index.js` and starts the auth server on port 4000.

5. Open the app in the browser (served by the Express server):

   - Login/Signup: http://localhost:4000/login.html
   - Dashboard:    http://localhost:4000/index.html

Environment variables (frontend/.env)
- PORT=4000
- MONGODB_URL=your_mongodb_connection_string
- CORS_ORIGIN=http://localhost:4000
- ACCESS_TOKEN_SECRET=some_random_secret
- ACCESS_TOKEN_EXPIRY=1d
- REFRESH_TOKEN_SECRET=another_secret
- REFRESH_TOKEN_EXPIRY=7d

Troubleshooting
- "Could not connect to any servers" — whitelist your IP in MongoDB Atlas (Network Access)
- "secretOrPrivateKey must have a value" — set the JWT secrets in `.env`
- Static files not loading — ensure Express serves the `frontend` directory and open `http://localhost:4000/login.html`
- If `nodemon` keeps restarting multiple times, run `npm start` from project root (script runs `cd frontend && nodemon index.js`) to isolate the service

Project structure (important files)
- `backend.py` — Flask app that hosts endpoints `/detect-image` and `/detect-video`
- `temple_structure_analyzer.py` — large analysis pipeline for videos
- `integratedimgdetection.py` — image detection helpers (feature matching & segmentation)
- `frontend/` — Express auth server that serves static frontend files + MongoDB auth controllers
  - `frontend/app.js`, `frontend/index.js`, `frontend/models/user.model.js`, `frontend/controllers/user.controller.js`, `frontend/routes/user.routes.js`
- `frontend/login.html` & `frontend/index.html` — minimal UI, wired to `frontend/main.js`

Security & next steps
- Do NOT use the development servers in production. Use proper production servers and HTTPS.
- Replace hard-coded JWT secrets with a secrets manager for production.
- Add input validation, rate-limiting and CSRF protection as needed.

License & credits
- Make sure to check model licenses for YOLO/pytorch if you redistribute.
- Built for research & educational purposes.

---
Made with ❤️ — TempleVision
