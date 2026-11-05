import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useWebcam } from '../src/hooks/useWebcam';
import { useFrameTracker } from '../src/hooks/useFrameTracker';
import { trackingClient } from '../src/services/trackingService';

describe('Webcam & Frame Tracker Lifecycle Safeguards', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  it('useWebcam handles permission denial cleanly without leaving dangling state', async () => {
    const notAllowedError = new Error('Permission denied');
    notAllowedError.name = 'NotAllowedError';

    navigator.mediaDevices = {
      getUserMedia: vi.fn().mockRejectedValue(notAllowedError),
    };

    const { result } = renderHook(() => useWebcam());

    await act(async () => {
      try {
        await result.current.startCamera();
      } catch (err) {
        // Expected
      }
    });

    expect(result.current.stream).toBeNull();
    expect(result.current.error).toContain('Camera permission was denied');
  });

  it('useWebcam handles missing camera hardware gracefully', async () => {
    const notFoundError = new Error('No device');
    notFoundError.name = 'NotFoundError';

    navigator.mediaDevices = {
      getUserMedia: vi.fn().mockRejectedValue(notFoundError),
    };

    const { result } = renderHook(() => useWebcam());

    await act(async () => {
      try {
        await result.current.startCamera();
      } catch (err) {
        // Expected
      }
    });

    expect(result.current.stream).toBeNull();
    expect(result.current.error).toContain('No camera device found');
  });

  it('useWebcam stops all media tracks immediately on stopCamera', async () => {
    const mockTrack1 = { stop: vi.fn() };
    const mockTrack2 = { stop: vi.fn() };
    const mockStream = {
      getTracks: vi.fn().mockReturnValue([mockTrack1, mockTrack2]),
    };

    navigator.mediaDevices = {
      getUserMedia: vi.fn().mockResolvedValue(mockStream),
    };

    const { result } = renderHook(() => useWebcam());

    await act(async () => {
      await result.current.startCamera();
    });

    expect(result.current.stream).toBe(mockStream);

    act(() => {
      result.current.stopCamera();
    });

    expect(mockTrack1.stop).toHaveBeenCalledTimes(1);
    expect(mockTrack2.stop).toHaveBeenCalledTimes(1);
    expect(result.current.stream).toBeNull();
  });

  it('useFrameTracker enforces single-flight and drops ticks when upload is in-flight', async () => {
    vi.useFakeTimers();

    // Mock trackingClient methods
    const createSessionSpy = vi
      .spyOn(trackingClient, 'createSession')
      .mockResolvedValue({ session_id: 'trk_sess_test' });

    // Mock a slow processFrame (latency > 200ms)
    let resolveFirstFrame;
    const slowUploadPromise = new Promise((resolve) => {
      resolveFirstFrame = resolve;
    });

    const processFrameSpy = vi
      .spyOn(trackingClient, 'processFrame')
      .mockImplementation(() => slowUploadPromise);

    const endSessionSpy = vi
      .spyOn(trackingClient, 'endSession')
      .mockResolvedValue({ session_id: 'trk_sess_test' });

    const mockVideo = {
      readyState: 4,
      videoWidth: 640,
      videoHeight: 480,
      paused: false,
      ended: false,
    };
    const videoRef = { current: mockVideo };

    // Mock offscreen canvas toBlob
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
      drawImage: vi.fn(),
    });
    HTMLCanvasElement.prototype.toBlob = vi.fn((cb) => {
      cb(new Blob(['fake'], { type: 'image/jpeg' }));
    });

    const { result } = renderHook(() =>
      useFrameTracker({
        videoRef,
        isCameraReady: true,
        token: 'valid-token',
        classSectionId: 1,
        targetFps: 5,
      })
    );

    // 1. Start tracking session
    await act(async () => {
      await result.current.startTrackingSession();
    });

    expect(createSessionSpy).toHaveBeenCalledWith(1);
    expect(result.current.isTracking).toBe(true);

    // Frame 0 started
    expect(processFrameSpy).toHaveBeenCalledTimes(1);
    expect(processFrameSpy.mock.calls[0][1]).toBe(0); // frame_seq = 0

    // Advance timer by 200ms while Frame 0 is still in flight
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    // Mandatory safeguard check: ProcessFrame MUST NOT have been called again (single-flight lock)
    expect(processFrameSpy).toHaveBeenCalledTimes(1);
    expect(result.current.metrics.framesDropped).toBe(1);

    // Advance timer again by 200ms
    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    expect(processFrameSpy).toHaveBeenCalledTimes(1);
    expect(result.current.metrics.framesDropped).toBe(2);

    // Now complete Frame 0
    await act(async () => {
      resolveFirstFrame({
        session_id: 'trk_sess_test',
        frame_seq: 0,
        active_tracks_count: 1,
        confirmed_tracks_count: 0,
        tracks: [
          {
            track_id: 'TRK-0001',
            state: 'tentative',
            bounding_box: { x: 50, y: 50, width: 100, height: 100 },
            detection_confidence: 0.95,
            total_observations: 1,
            consecutive_misses: 0,
            first_seen_frame: 0,
            last_seen_frame: 0,
            is_newly_created: true,
            is_recovered: false,
            is_ambiguous: false,
          },
        ],
      });
    });

    // Advance timer to trigger next frame tick
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    // Now Frame 1 should have been dispatched with strictly monotonic frame_seq = 1
    expect(processFrameSpy).toHaveBeenCalledTimes(2);
    expect(processFrameSpy.mock.calls[1][1]).toBe(1); // frame_seq = 1

    // Teardown
    await act(async () => {
      await result.current.endTrackingSession();
    });
    expect(endSessionSpy).toHaveBeenCalledWith('trk_sess_test');
    expect(result.current.isTracking).toBe(false);

    vi.useRealTimers();
  });
});
