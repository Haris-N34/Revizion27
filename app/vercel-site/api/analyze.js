const backend = 'https://textbooks-balloon-metres-music.trycloudflare.com';
export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({error:'Use POST'});
  try {
    const response = await fetch(`${backend}/api/analyze`, {
      method:'POST', headers:{'Content-Type':'application/json',Origin:backend},
      body:JSON.stringify(req.body), signal:AbortSignal.timeout(50000)
    });
    const text = await response.text();
    if (!response.headers.get('content-type')?.includes('application/json')) {
      return res.status(503).json({error:'The detection laptop is offline. Your camera preview still works.'});
    }
    res.status(response.status).setHeader('Content-Type','application/json').end(text);
  } catch {
    res.status(503).json({error:'Cannot reach the detection laptop. Your camera preview still works.'});
  }
}
