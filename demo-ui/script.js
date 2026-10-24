const $ = (id) => document.getElementById(id);
const root = document.documentElement;
function preference(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(`revizion-${key}`) === 'true';
    localStorage.setItem(`revizion-${key}`, String(value));
  } catch { return false; }
}
for (const [id, className] of [['text-toggle', 'large-text'], ['contrast-toggle', 'high-contrast']]) {
  const button = $(id);
  const apply = (enabled) => { root.classList.toggle(className, enabled); button.setAttribute('aria-pressed', String(enabled)); };
  apply(preference(className));
  button.addEventListener('click', () => { const enabled = !root.classList.contains(className); apply(enabled); preference(className, enabled); });
}
let stage = 0;
let active = false;
let object = 'can';
let missing = false;
const display = document.querySelector('.demo-display');
const canArt = $('target-object').querySelector('svg').innerHTML;
const objectArt = {
  can: canArt,
  cup: '<path d="M13 30h46v48c0 24-46 24-46 0Z" fill="#e65a29" stroke="#292c25" stroke-width="2"/><path d="M59 40c25-5 25 35 0 30" fill="none" stroke="#292c25" stroke-width="6"/><ellipse cx="36" cy="30" rx="23" ry="7" fill="#f5f3eb" stroke="#292c25" stroke-width="2"/>',
  bottle: '<path d="M30 17h20v20l13 17v39c0 10-46 10-46 0V54l13-17Z" fill="#7e9567" stroke="#292c25" stroke-width="2"/><rect x="28" y="10" width="24" height="12" rx="3" fill="#292c25"/><path d="M18 60h44v23H18Z" fill="#f5f3eb"/><text x="40" y="78" text-anchor="middle" fill="#292c25" font-size="19" font-weight="bold">r.</text>'
};
function speak(text) {
  if (!$('spoken').checked || !('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const cue = new SpeechSynthesisUtterance(text);
  cue.rate = 0.95;
  speechSynthesis.speak(cue);
}
function render(announce = true) {
  const states = [
    ['↗', 'LET’S BEGIN', 'A little guidance starts here.', 'Choose an object, then start your reach.', 'Vibration cue: resting'],
    ['◎', '01 / LOOKING', `Looking for your ${object}.`, 'The camera would scan the scene for the object you named.', 'Vibration cue: resting while scanning'],
    ['→', '02 / OBJECT FOUND', `${object[0].toUpperCase() + object.slice(1)} found. Move right.`, 'A vibration on the right side of your wrist points the way.', 'Vibration cue: right motor'],
    ['↑', '03 / GETTING CLOSER', 'A little further forward.', 'Your hand is getting closer. The guidance updates as you move.', 'Vibration cue: forward motor'],
    ['✓', '04 / WITHIN REACH', 'You’re there. Ready to grab.', `Your hand has reached the ${object}. A distinct vibration signals arrival.`, 'Vibration cue: all motors · grab pattern']
  ];
  const state = missing && stage === 2 ? ['—', '02 / OBJECT NOT FOUND', `I can’t find a ${object}.`, 'The scan has ended. Guidance stops instead of guessing.', 'Vibration cue: all motors off'] : states[stage];
  $('guidance-arrow').textContent = state[0];
  $('guidance-label').textContent = state[1];
  $('guidance-title').textContent = state[2];
  $('guidance-description').textContent = state[3];
  $('haptic-description').textContent = state[4];
  display.dataset.stage = String(stage);
  display.dataset.missing = String(missing);
  const done = stage === 4 || (missing && stage === 2);
  $('stage-count').textContent = stage === 0 ? 'READY WHEN YOU ARE' : done ? 'EXPERIENCE COMPLETE' : `STEP ${stage} OF ${missing ? 2 : 4}`;
  $('next-step').disabled = !active || done;
  $('start-demo').innerHTML = active ? 'Start again <span aria-hidden="true">↗</span>' : 'Start my reach <span aria-hidden="true">↗</span>';
  if (announce) speak(`${state[2]} ${state[3]} ${state[4]}`);
}
function reset(announce = false) {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  stage = 0; active = false;
  object = document.querySelector('input[name="object"]:checked').value;
  missing = $('missing').checked;
  $('object-label').textContent = object.toUpperCase();
  $('target-object').querySelector('svg').innerHTML = objectArt[object];
  render(announce);
}
$('start-demo').addEventListener('click', () => { reset(); active = true; stage = 1; render(); });
$('next-step').addEventListener('click', () => { if (active && stage < 4 && !(missing && stage === 2)) { stage += 1; render(); } });
$('reset-demo').addEventListener('click', () => reset(true));
document.querySelectorAll('input[name="object"], #missing').forEach(input => input.addEventListener('change', () => reset(true)));
$('spoken').addEventListener('change', () => {
  if (!$('spoken').checked && 'speechSynthesis' in window) speechSynthesis.cancel();
  else speak(`${$('guidance-title').textContent} ${$('guidance-description').textContent}`);
});
if (!('speechSynthesis' in window)) { $('spoken').disabled = true; $('speech-note').textContent = '(unavailable in this browser)'; }
window.addEventListener('pagehide', () => { if ('speechSynthesis' in window) speechSynthesis.cancel(); });
reset();
