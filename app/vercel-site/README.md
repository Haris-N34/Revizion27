# REVISION on Vercel

Production: https://revision-live-guidance.vercel.app
Deploy from this directory with `npx vercel --prod` after signing in to Vercel.
Enable camera once to grant browser permission. Automatic selection prefers DroidCam,
then the browser default. An explicit choice is remembered only on that browser.
Keep DroidCam running on the phone and Windows client; if busy, close other camera
apps or select Laptop + wristband to use the already-open laptop camera and Arduino.
Each tab uploads its own frames through `/api/analyze`; the laptop performs detection
and keeps hand tracking separate for up to four active tabs. Browser mode never drives the band.
Keep the laptop app and tunnel running. After a tunnel URL change, update `camera.js`
and `api/analyze.js` and redeploy. No frontend build is needed.
