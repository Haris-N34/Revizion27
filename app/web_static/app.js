const $ = (id) => document.getElementById(id);
let current = {}, supportedKey = '', requesting = false, connectionNotice = false;
const icons = {can:'▥', bottle:'♙', cup:'☕', 'cell phone':'▯', remote:'▤', mouse:'◒'};
const labels = {'cell phone':'Phone'};
const label = target => labels[target] || target.charAt(0).toUpperCase() + target.slice(1);
function notice(text) { $('notice').textContent = text || ''; $('notice').hidden = !text; }
async function action(name, payload = {}) {
  if (requesting) return;
  requesting = true;
  try {
    const response = await fetch(`/api/${name}`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not send your request.');
    notice('');
  } catch(error) { notice(error.message); }
  finally { requesting = false; }
}
function show(data) {
  current = data;
  const stale = data.frame_age_s !== null && data.frame_age_s > 3;
  const ready = data.ready && !stale;
  $('status').textContent = stale ? 'Camera feed interrupted' : data.status;
  $('phase-pill').textContent = data.listening === 'recording' ? 'LISTENING' : data.phase || 'CONNECTING';
  $('status-bar').classList.toggle('arrived', data.phase === 'DONE');
  $('status-symbol').textContent = data.phase === 'DONE' ? '✓' : '◉';
  $('camera-label').textContent = data.demo ? 'Demo camera · simulated feed' : ready ? 'Connected camera · live' : 'Waiting for camera';
  $('fps').textContent = `${data.fps || '—'} FPS`;
  $('target-value').textContent = data.target ? label(data.target) : 'None selected';
  $('hand-value').textContent = data.hand ? 'Hand visible' : 'No hand in view';
  $('band-value').textContent = data.arduino ? (data.direction || 'Connected · idle') : data.ready ? 'Demo / mock band' : 'Connecting';
  $('talk').disabled = !ready || !data.voice || ['recording','transcribing'].includes(data.listening);
  $('talk').classList.toggle('recording', data.listening === 'recording');
  $('talk-label').textContent = data.listening === 'recording' ? 'Listening…' : data.listening === 'transcribing' ? 'Recognizing…' : 'Press to talk';
  $('voice-help').textContent = data.ready && !data.voice ? 'Voice is off for this session. Choose a target above.' : 'Press the wristband button, say “find a can,” then pause.';
  $('stop').disabled = !data.ready && data.phase !== 'RECONNECTING';
  $('rotate').disabled = !ready || ['recording','transcribing'].includes(data.listening);
  $('rotate').textContent = `↻ Rotate camera · ${data.rotation || 0}°`;
  if (data.events?.length) $('feedback').textContent = data.events[0];
  if (data.error || stale) {
    notice(data.error || 'Camera feed interrupted. Guidance is paused while REVISION reconnects automatically.');
    connectionNotice = true;
  } else if (connectionNotice && ready) {
    notice('');
    connectionNotice = false;
  }
  const placeholder = $('camera-placeholder');
  placeholder.style.display = data.frame_age_s !== null && !stale ? 'none' : 'flex';
  if (data.error) { placeholder.querySelector('strong').textContent = 'Let’s reconnect your camera'; placeholder.querySelector('span:last-child').textContent = data.error; }
  const key = JSON.stringify(data.supported || []);
  if (key !== supportedKey) {
    supportedKey = key;
    $('targets').replaceChildren();
    for (const target of data.supported || []) {
      const button = document.createElement('button');
      button.className = 'target'; button.dataset.target = target;
      const icon = document.createElement('span'); icon.className = 'icon'; icon.setAttribute('aria-hidden','true'); icon.textContent = icons[target] || '◎';
      const text = document.createElement('span'); text.textContent = label(target);
      button.append(icon,text); button.onclick = () => action('target',{target});
      $('targets').append(button);
    }
  }
  for (const button of $('targets').children) {
    button.classList.toggle('active',button.dataset.target === data.target);
    button.setAttribute('aria-pressed',button.dataset.target === data.target ? 'true':'false');
    button.disabled = !ready || ['recording','transcribing'].includes(data.listening);
  }
}
async function poll() {
  try {
    const response = await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(3000)});
    if (!response.ok) throw new Error();
    show(await response.json());
  } catch {
    notice('Lost connection to REVISION. Retrying automatically.');
    connectionNotice = true;
    $('status').textContent = 'Disconnected'; $('talk').disabled = true; $('stop').disabled = true;
    $('rotate').disabled = true;
    for (const button of $('targets').children) button.disabled = true;
  }
  setTimeout(poll,500);
}
$('talk').onclick = () => action('listen');
$('stop').onclick = () => action('stop');
$('rotate').onclick = () => action('rotate',{rotation:((current.rotation || 0)+90)%360});
document.addEventListener('keydown',event => {
  if (event.repeat || ['INPUT','TEXTAREA','SELECT','BUTTON'].includes(event.target.tagName)) return;
  if (event.code === 'Escape' && !$('stop').disabled) action('stop');
});
document.addEventListener('visibilitychange',()=>{if(document.hidden && (current.ready || current.phase === 'RECONNECTING')) action('stop');});
poll();
