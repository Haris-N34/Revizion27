import test from 'node:test';
import assert from 'node:assert/strict';
import {preferredCamera, cameraError} from '../vercel-site/camera-choice.mjs';
const devices = [
  {kind:'videoinput',deviceId:'default',label:'Integrated webcam'},
  {kind:'videoinput',deviceId:'phone',label:'DroidCam Video'},
  {kind:'videoinput',deviceId:'usb',label:'USB webcam'},
  {kind:'audioinput',deviceId:'mic',label:'DroidCam Audio'}
];
test('prefers DroidCam after permission reveals labels',()=>assert.equal(preferredCamera(devices,null).deviceId,'phone'));
test('honors a camera explicitly selected on this browser',()=>assert.equal(preferredCamera(devices,'usb').deviceId,'usb'));
test('other devices use the browser default without a forced index',()=>assert.equal(preferredCamera(devices.slice(0,1),null),null));
test('removed preferences fall back to DroidCam or default',()=>{
  assert.equal(preferredCamera(devices,'removed').deviceId,'phone');
  assert.equal(preferredCamera(devices.slice(0,1),'removed'),null);
});
test('permission denial and busy camera have actionable messages',()=>{
  assert.match(cameraError({name:'NotAllowedError'}),/site settings/);
  assert.match(cameraError({name:'NotReadableError'}),/Laptop \+ wristband/);
});
