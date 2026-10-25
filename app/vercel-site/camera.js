import {preferredCamera, cameraError} from './camera-choice.mjs';
const $ = id => document.getElementById(id);
const laptopURL = 'https://textbooks-balloon-metres-music.trycloudflare.com';
const session = crypto.randomUUID();
let stream = null, target = null, angle = 0, version = 0, result = null, resultAt = 0;
let starting = false, raf = 0, request = null, timer = 0;
const video = $('video'), preview = $('preview'), capture = document.createElement('canvas');
const ctx = preview.getContext('2d'), captureCtx = capture.getContext('2d');
const read = key => {try{return localStorage.getItem(key);}catch{return null;}};
const save = (key,value) => {try{localStorage.setItem(key,value);}catch{}};
const notice = text => {$('error').textContent=text; $('error').hidden=!text;};
const status = (text,detail) => {$('status').textContent=text; $('detail').textContent=detail;};
function cancelInference() {
  version++; result=null; clearTimeout(timer); request?.abort(); request=null;
}
function stopCamera() {
  cancelInference(); cancelAnimationFrame(raf);
  stream?.getTracks().forEach(track=>track.stop()); stream=null; video.srcObject=null;
  preview.hidden=true; $('placeholder').hidden=false;
  $('stop').disabled=true; $('rotate').disabled=true;
  $('camera-name').textContent='Camera stopped';
}
async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter(device=>device.kind==='videoinput');
  const selected = read('revision.camera') || '';
  $('camera').replaceChildren(new Option('Automatic · DroidCam or default webcam',''));
  cameras.forEach((device,index)=>$('camera').add(new Option(device.label || `Camera ${index+1}`,device.deviceId)));
  $('camera').value=cameras.some(camera=>camera.deviceId===selected)?selected:'';
  $('camera').disabled=false;
  return devices;
}
async function startCamera() {
  if(starting) return;
  starting=true; $('start').disabled=true; $('reconnect').disabled=true;
  stopCamera(); notice('');
  status('Connecting camera…','Allow camera access when your browser asks.');
  try {
    if(!navigator.mediaDevices?.getUserMedia) throw {name:'NotFoundError'};
    // Use known labels/preferences first; the initial permission unlocks hidden labels.
    const known=preferredCamera(await navigator.mediaDevices.enumerateDevices(),read('revision.camera'));
    stream=await navigator.mediaDevices.getUserMedia({video:{...(known?{deviceId:{exact:known.deviceId}}:{}),width:{ideal:640},height:{ideal:480}},audio:false});
    const devices=await listCameras();
    const preferred=preferredCamera(devices,read('revision.camera'));
    const current=stream.getVideoTracks()[0].getSettings().deviceId;
    if(preferred && preferred.deviceId!==current) {
      stream.getTracks().forEach(track=>track.stop());
      stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:{exact:preferred.deviceId},width:{ideal:640},height:{ideal:480}},audio:false});
    }
    if($('mode').value!=='device'){stopCamera();return;}
    video.srcObject=stream; await video.play();
    $('camera-name').textContent=stream.getVideoTracks()[0].label || 'Default webcam';
    stream.getVideoTracks()[0].onended=()=>{stopCamera();notice('Camera disconnected. Reconnect it, then press Enable camera.');status('Camera disconnected','Your target is saved. Reconnect to continue.');};
    $('placeholder').hidden=true; preview.hidden=false;
    $('stop').disabled=false; $('rotate').disabled=false;
    status('Camera connected',target?`Looking for ${target}.`:'Choose an object to start detection.');
    draw(); scheduleInference();
  } catch(error) {stopCamera();notice(cameraError(error));status('Camera needs attention','Choose a camera or reconnect, then try again.');}
  finally {starting=false;$('start').disabled=false;$('reconnect').disabled=false;}
}
function drawFrame(canvas,context) {
  const width=video.videoWidth, height=video.videoHeight;
  if(!width || !height) return false;
  const scale=Math.min(1,640/Math.max(width,height));
  const w=Math.round(width*scale),h=Math.round(height*scale);
  const outW=angle%180?h:w,outH=angle%180?w:h;
  if(canvas.width!==outW || canvas.height!==outH){canvas.width=outW;canvas.height=outH;}
  context.save();context.translate(outW/2,outH/2);context.rotate(angle*Math.PI/180);
  context.drawImage(video,-w/2,-h/2,w,h);context.restore();return true;
}
function draw() {
  if(!stream)return;
  drawFrame(preview,ctx);
  if(result && performance.now()-resultAt<1500 && result.width===preview.width && result.height===preview.height) {
    ctx.lineWidth=3;ctx.strokeStyle='#d4ed8b';ctx.fillStyle='#d4ed8b';ctx.font='16px system-ui';
    for(const det of result.detections){ctx.strokeRect(det.x1,det.y1,det.x2-det.x1,det.y2-det.y1);ctx.fillText(`${det.label} ${Math.round(det.confidence*100)}%`,det.x1,Math.max(18,det.y1-8));}
    if(result.hand){ctx.fillStyle='#ffb88a';ctx.beginPath();ctx.arc(result.hand.x,result.hand.y,7,0,Math.PI*2);ctx.fill();}
  }
  raf=requestAnimationFrame(draw);
}
function scheduleInference(){clearTimeout(timer);if(stream&&target&&!document.hidden)timer=setTimeout(analyze,350);}
async function analyze() {
  if(!stream||!target||document.hidden||request)return;
  const generation=version, chosen=target;
  if(!drawFrame(capture,captureCtx)){scheduleInference();return;}
  request=new AbortController();
  try {
    const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},signal:request.signal,
      body:JSON.stringify({session,target:chosen,image:capture.toDataURL('image/jpeg',0.75).split(',')[1],mirror:$('mirror').checked})});
    const data=await response.json();
    if(generation!==version)return;
    if(!response.ok)throw new Error(data.error||'Detection unavailable. Retrying…');
    result=data;resultAt=performance.now();notice('');
    if(!data.detections.length)status(`Looking for ${chosen}`, 'Keep the object in view.');
    else if(!data.hand)status(`${chosen} found`, 'Bring your hand into view for guidance.');
    else status(data.command?.direction==='ARRIVED'?'Hand aligned with target':`Move ${data.command?.direction.toLowerCase()}`,`Guiding your hand to ${chosen}.`);
    $('connection').textContent='Detection connected · on-screen guidance';
  } catch(error) {
    if(generation===version && error.name!=='AbortError'){
      result=null;notice(error.message);status('Camera live · detection reconnecting','Your webcam is working. Waiting for the detection laptop.');
      $('connection').textContent='Camera live · detection unavailable';
    }
  } finally {if(generation===version){request=null;scheduleInference();}}
}
for(const name of ['can','bottle','cup','cell phone','remote','mouse']) {
  const button=document.createElement('button');button.textContent=name==='cell phone'?'Phone':name[0].toUpperCase()+name.slice(1);
  button.onclick=()=>{cancelInference();target=name;for(const other of $('targets').children){other.classList.toggle('active',other===button);other.setAttribute('aria-pressed',String(other===button));}status(`Looking for ${name}`,stream?'Keep the object and your hand in view.':'Enable your camera to begin.');scheduleInference();};
  button.setAttribute('aria-pressed','false');$('targets').append(button);
}
$('start').onclick=startCamera;$('reconnect').onclick=startCamera;
$('stop').onclick=()=>{stopCamera();status('Camera stopped','Press Enable camera to continue.');};
$('camera').onchange=()=>{save('revision.camera',$('camera').value);startCamera();};
$('rotate').onclick=()=>{cancelInference();angle=(angle+90)%360;$('rotate').textContent=`↻ Rotate ${angle}°`;scheduleInference();};
$('mirror').onchange=()=>{cancelInference();scheduleInference();};
$('mode').onchange=()=>{const laptop=$('mode').value==='laptop';stopCamera();$('device-mode').hidden=laptop;$('laptop-mode').hidden=!laptop;$('laptop').src=laptop?laptopURL:'about:blank';notice('');};
navigator.mediaDevices?.addEventListener('devicechange',async()=>{try{await listCameras();}catch{}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)cancelInference();else scheduleInference();});
window.addEventListener('pagehide',stopCamera);
