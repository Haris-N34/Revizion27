export function preferredCamera(devices, saved) {
  const cameras = devices.filter(device => device.kind === 'videoinput');
  return cameras.find(camera => saved && camera.deviceId === saved)
    || cameras.find(camera => /droidcam/i.test(camera.label))
    || null; // Let getUserMedia choose the browser default.
}

export function cameraError(error) {
  return ({NotAllowedError:'Camera access is blocked. Allow the camera in your browser’s site settings, then reconnect.',
    NotFoundError:'No camera was found. Connect a webcam or start DroidCam on your phone and computer.',
    NotReadableError:'The camera is busy. Close other camera apps, or choose Laptop + wristband if REVISION is already using DroidCam.',
    OverconstrainedError:'That camera is no longer available. Choose Automatic and reconnect.'})[error.name]
    || 'Could not start the camera. Check its connection and try again.';
}
