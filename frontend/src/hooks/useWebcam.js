import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Custom hook to safely manage browser webcam stream lifecycle.
 * Guarantees all MediaStreamTracks are terminated when stopped or unmounted.
 */
export function useWebcam() {
  const [stream, setStream] = useState(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks();
      tracks.forEach((track) => {
        try {
          track.stop();
        } catch (e) {
          console.warn('Error stopping track:', e);
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStream(null);
    setIsReady(false);
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    setError(null);

    if (!navigator?.mediaDevices?.getUserMedia) {
      const msg = 'Camera API is not supported in this browser or insecure context (HTTPS/localhost required).';
      setError(msg);
      throw new Error(msg);
    }

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = mediaStream;
      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      return mediaStream;
    } catch (err) {
      let friendlyMsg = 'Failed to access camera.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        friendlyMsg = 'Camera permission was denied. Please allow camera access in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        friendlyMsg = 'No camera device found on this system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        friendlyMsg = 'Camera is currently in use by another application or tab.';
      } else if (err.message) {
        friendlyMsg = err.message;
      }
      stopCamera();
      setError(friendlyMsg);
      throw new Error(friendlyMsg);
    }
  }, [stopCamera]);

  // Clean up all tracks on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    stream,
    isReady,
    setIsReady,
    error,
    startCamera,
    stopCamera,
  };
}
