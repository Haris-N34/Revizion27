import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

test('camera reconnect clears old warnings and keeps Stop available', () => {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent:'', hidden:true, disabled:false, children:[], style:{},
      classList:{toggle(){}}, querySelector: selector => element(id + selector),
      replaceChildren(){this.children=[];}
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    document:{getElementById:element,addEventListener(){}},
    setTimeout(){},
  });
  const source = readFileSync(new URL('../web_static/app.js', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/^poll\(\);\s*$/m, ''), context);
  const data = {ready:false,phase:'RECONNECTING',error:'Reconnecting DroidCam',
    frame_age_s:null,supported:[],events:[],voice:true,arduino:true};
  context.show(data);
  assert.equal(element('notice').hidden,false);
  assert.equal(element('stop').disabled,false);
  context.show({...data,ready:true,phase:'IDLE',error:null,frame_age_s:0});
  assert.equal(element('notice').hidden,true);
  assert.equal(element('notice').textContent,'');
  assert.equal(element('camera-placeholder').style.display,'none');
});
