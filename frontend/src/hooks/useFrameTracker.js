import { useState, useRef, useCallback, useEffect } from 'react';
import { trackingClient } from '../services/trackingService';

/**
 * Custom hook to control frame capture loop (~5 FPS), ensure strictly monotonic frame sequence,
 * drop frames when an upload is in-flight (zero backlog), and handle session errors gracefully.
 */
export function useFrameTracker({ videoRef, isCameraReady, token, classSectionId, targetFps = 5 }) {
  const [isTracking, setIsTracking] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [metrics, setMetrics] = useState({
    fps: 0,
    framesProcessed: 0,
    framesDropped: 0,
    activeTracksCount: 0,
    confirmedTracksCount: 0,
    lastLatencyMs: 0,
  });
  const [error, setError] = useState(null);

  const isTrackingRef = useRef(false);
  const isProcessingRef = useRef(false);
  const sessionIdRef = useRef(null);
  const nextFrameSeqRef = useRef(0);
  const loopTimeoutRef = useRef(null);
  const offscreenCanvasRef = useRef(null);

  // Performance metrics tracking refs
  const frameCountRef = useRef(0);
  const dropCountRef = useRef(0);
  const fpsWindowStartRef = useRef(performance.now());

  // Keep client token synchronized
  useEffect(() => {
    trackingClient.setToken(token);
  }, [token]);

  const endTrackingSession = useCallback(async () => {
    isTrackingRef.current = false;
    setIsTracking(false);

    if (loopTimeoutRef.current) {
      clearTimeout(loopTimeoutRef.current);
      loopTimeoutRef.current = null;
    }

    const currentSessionId = sessionIdRef.current;
    sessionIdRef.current = null;
    setSessionId(null);
    setTracks([]);
    isProcessingRef.current = false;

    if (currentSessionId) {
      try {
        await trackingClient.endSession(currentSessionId);
      } catch (err) {
        console.warn('Session teardown warning:', err.message);
      }
    }
  }, []);

  const captureFrameBlob = useCallback((video) => {
    if (!video || video.readyState < 2) return null;

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
    }
    const canvas = offscreenCanvasRef.current;
    const w = video.videoWidth;
    const h = video.videoHeight;

    if (w === 0 || h === 0) return null;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);

    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob);
        },
        'image/jpeg',
        0.85
      );
    });
  }, []);

  const runFrameTick = useCallback(async () => {
    if (!isTrackingRef.current || !sessionIdRef.current) {
      return;
    }

    const intervalMs = Math.round(1000 / targetFps);

    // 1. Mandatory Safeguard: Single-flight lock.
    // If previous upload is still in flight, DROP this tick immediately.
    if (isProcessingRef.current) {
      dropCountRef.current += 1;
      setMetrics((m) => ({ ...m, framesDropped: dropCountRef.current }));
      loopTimeoutRef.current = setTimeout(runFrameTick, intervalMs);
      return;
    }

    const video = videoRef.current;
    if (!video || video.paused || video.ended) {
      loopTimeoutRef.current = setTimeout(runFrameTick, intervalMs);
      return;
    }

    // 2. Extract frame from video element
    const blob = await captureFrameBlob(video);
    if (!blob) {
      loopTimeoutRef.current = setTimeout(runFrameTick, intervalMs);
      return;
    }

    // 3. Mark in-flight lock and advance monotonic sequence
    isProcessingRef.current = true;
    const frameSeq = nextFrameSeqRef.current++;
    const timestamp = Date.now() / 1000.0;
    const startTime = performance.now();

    // Schedule next tick after target interval regardless of whether this frame is finished,
    // so that subsequent ticks can detect in-flight processing and drop frames.
    loopTimeoutRef.current = setTimeout(runFrameTick, intervalMs);

    try {
      const result = await trackingClient.processFrame(
        sessionIdRef.current,
        frameSeq,
        timestamp,
        blob,
        8000 // 8-second timeout
      );

      const elapsed = performance.now() - startTime;
      frameCountRef.current += 1;

      // Update FPS calculation over rolling window
      const now = performance.now();
      const windowDuration = (now - fpsWindowStartRef.current) / 1000.0;
      let currentFps = metrics.fps;
      if (windowDuration >= 1.0) {
        currentFps = Math.round((frameCountRef.current / windowDuration) * 10) / 10;
        frameCountRef.current = 0;
        fpsWindowStartRef.current = now;
      }

      setTracks(result.tracks || []);
      setMetrics((m) => ({
        ...m,
        fps: currentFps,
        framesProcessed: frameSeq + 1,
        framesDropped: dropCountRef.current,
        activeTracksCount: result.active_tracks_count,
        confirmedTracksCount: result.confirmed_tracks_count,
        lastLatencyMs: Math.round(elapsed),
      }));
      setError(null);
    } catch (err) {
      console.error(`Frame ${frameSeq} processing failed:`, err);
      // If sequence state is uncertain or session expired (404), halt tracking
      if (err.message.includes('strictly greater') || err.message.includes('404') || err.message.includes('timed out')) {
        setError(`Tracking halted due to error: ${err.message}. Please restart session.`);
        isTrackingRef.current = false;
        setIsTracking(false);
        return;
      } else {
        setError(err.message);
      }
    } finally {
      isProcessingRef.current = false;
    }
  }, [targetFps, videoRef, captureFrameBlob, metrics.fps]);

  const startTrackingSession = useCallback(async () => {
    if (!token) {
      throw new Error('Please enter a valid Faculty/Admin JWT Access Token.');
    }
    if (!classSectionId || Number(classSectionId) <= 0) {
      throw new Error('Please provide a valid Class Section ID.');
    }

    setError(null);
    nextFrameSeqRef.current = 0;
    frameCountRef.current = 0;
    dropCountRef.current = 0;
    fpsWindowStartRef.current = performance.now();

    // 1. Create in-memory backend session
    const sessionRes = await trackingClient.createSession(classSectionId);
    const sid = sessionRes.session_id;

    sessionIdRef.current = sid;
    setSessionId(sid);
    isTrackingRef.current = true;
    setIsTracking(true);

    // 2. Start loop
    runFrameTick();
  }, [token, classSectionId, runFrameTick]);

  // Cleanup on unmount or navigation
  useEffect(() => {
    return () => {
      if (sessionIdRef.current) {
        // Best-effort cleanup while component unmounts
        trackingClient.endSession(sessionIdRef.current).catch(() => {});
      }
      if (loopTimeoutRef.current) {
        clearTimeout(loopTimeoutRef.current);
      }
    };
  }, []);

  return {
    isTracking,
    sessionId,
    tracks,
    metrics,
    error,
    startTrackingSession,
    endTrackingSession,
  };
}
