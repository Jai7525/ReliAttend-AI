import React, { useState } from 'react';
import { useWebcam } from '../../hooks/useWebcam';
import { useFrameTracker } from '../../hooks/useFrameTracker';
import { VideoOverlayCanvas } from './VideoOverlayCanvas';

export function TrackingCameraView() {
  const [token, setToken] = useState('');
  const [classSectionId, setClassSectionId] = useState('1');
  const [uiError, setUiError] = useState(null);

  const { videoRef, stream, isReady, setIsReady, error: cameraError, startCamera, stopCamera } = useWebcam();

  const {
    isTracking,
    sessionId,
    tracks,
    metrics,
    error: trackerError,
    startTrackingSession,
    endTrackingSession,
  } = useFrameTracker({
    videoRef,
    isCameraReady: isReady,
    token,
    classSectionId,
    targetFps: 5,
  });

  const handleStart = async () => {
    setUiError(null);
    try {
      // 1. Start browser camera (triggers permission prompt)
      await startCamera();
      // 2. Start tracking session
      await startTrackingSession();
    } catch (err) {
      setUiError(err.message);
      stopCamera();
      await endTrackingSession();
    }
  };

  const handleStop = async () => {
    // 1. Stop camera tracks immediately
    stopCamera();
    // 2. Terminate session and cleanup state
    await endTrackingSession();
  };

  const displayError = uiError || cameraError || trackerError;

  return (
    <div style={styles.container}>
      {/* Header */}
      <header style={styles.header}>
        <h1 style={styles.title}>ReliAttend AI — Multi-Face Tracking (Phase 9)</h1>
        <p style={styles.subtitle}>
          Real-time in-browser face tracking with Kalman filtering & IoU association. Zero raw biometric or video storage.
        </p>
      </header>

      {/* Dev Configuration Bar */}
      <section style={styles.configBar}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>
            JWT Access Token (Faculty/Admin):
            <input
              type="password"
              placeholder="Paste Bearer token from login..."
              value={token}
              onChange={(e) => setToken(e.target.value)}
              disabled={isTracking}
              style={styles.input}
            />
          </label>
        </div>

        <div style={styles.inputGroupSmall}>
          <label style={styles.label}>
            Class Section ID:
            <input
              type="number"
              min="1"
              value={classSectionId}
              onChange={(e) => setClassSectionId(e.target.value)}
              disabled={isTracking}
              style={styles.inputSmall}
            />
          </label>
        </div>

        <div style={styles.actionGroup}>
          {!isTracking ? (
            <button onClick={handleStart} style={styles.btnStart}>
              ▶ Start Tracking
            </button>
          ) : (
            <button onClick={handleStop} style={styles.btnStop}>
              ⏹ Stop Tracking
            </button>
          )}
        </div>
      </section>

      {/* Error Banner */}
      {displayError && (
        <div style={styles.errorBanner}>
          ⚠️ <strong>Error:</strong> {displayError}
        </div>
      )}

      {/* Main Grid: Camera Preview & Live Track Dashboard */}
      <div style={styles.mainGrid}>
        {/* Left: Video Preview with Overlay */}
        <div style={styles.videoCard}>
          <div style={styles.videoContainer}>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              onLoadedMetadata={() => setIsReady(true)}
              style={styles.video}
            />
            {isTracking && isReady && <VideoOverlayCanvas videoRef={videoRef} tracks={tracks} />}

            {!stream && (
              <div style={styles.videoPlaceholder}>
                <p>Camera is currently idle.</p>
                <p style={{ fontSize: '13px', color: '#6B7280' }}>Click "Start Tracking" to request camera permission.</p>
              </div>
            )}
          </div>

          {/* Real-time Telemetry Bar */}
          <div style={styles.telemetryBar}>
            <div>
              Status:{' '}
              <span style={{ color: isTracking ? '#10B981' : '#6B7280', fontWeight: 'bold' }}>
                {isTracking ? '● TRACKING ACTIVE' : '○ IDLE'}
              </span>
            </div>
            {sessionId && <div>Session: <code style={styles.code}>{sessionId}</code></div>}
            <div>FPS: <strong>{metrics.fps}</strong> (~5 target)</div>
            <div>Latency: <strong>{metrics.lastLatencyMs} ms</strong></div>
            <div>Dropped: <strong style={{ color: metrics.framesDropped > 0 ? '#F59E0B' : '#10B981' }}>{metrics.framesDropped}</strong></div>
          </div>
        </div>

        {/* Right: Active Tracks Panel */}
        <div style={styles.panelCard}>
          <h2 style={styles.panelTitle}>Active Track States ({metrics.activeTracksCount})</h2>
          <div style={styles.statsSummary}>
            <span style={styles.statBadgeGreen}>Confirmed: {metrics.confirmedTracksCount}</span>
            <span style={styles.statBadgeYellow}>Tentative: {metrics.activeTracksCount - metrics.confirmedTracksCount}</span>
          </div>

          <div style={styles.tracksList}>
            {tracks.length === 0 ? (
              <p style={styles.emptyText}>No active face tracks in current frame.</p>
            ) : (
              tracks.map((t) => (
                <div key={t.track_id} style={styles.trackCard}>
                  <div style={styles.trackCardHeader}>
                    <strong>{t.track_id}</strong>
                    <span style={getStateBadgeStyle(t.state, t.is_ambiguous)}>
                      {t.is_ambiguous ? 'AMBIGUOUS' : t.state.toUpperCase()}
                    </span>
                  </div>
                  <div style={styles.trackDetails}>
                    <div>Hits: {t.total_observations} | Misses: {t.consecutive_misses}</div>
                    <div>Confidence: {(t.detection_confidence * 100).toFixed(1)}%</div>
                    <div>Frames: #{t.first_seen_frame} – #{t.last_seen_frame}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function getStateBadgeStyle(state, isAmbiguous) {
  if (isAmbiguous) {
    return { backgroundColor: '#FEE2E2', color: '#B91C1C', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' };
  }
  switch (state) {
    case 'confirmed':
      return { backgroundColor: '#D1FAE5', color: '#047857', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' };
    case 'tentative':
      return { backgroundColor: '#FEF3C7', color: '#B45309', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' };
    case 'temporarily_lost':
      return { backgroundColor: '#FFEDD5', color: '#C2410C', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold' };
    default:
      return { backgroundColor: '#F3F4F6', color: '#4B5563', padding: '2px 8px', borderRadius: '4px', fontSize: '11px' };
  }
}

const styles = {
  container: {
    maxWidth: '1280px',
    margin: '0 auto',
    padding: '24px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color: '#111827',
  },
  header: {
    marginBottom: '20px',
    borderBottom: '1px solid #E5E7EB',
    paddingBottom: '16px',
  },
  title: {
    margin: '0 0 6px 0',
    fontSize: '24px',
    fontWeight: '700',
  },
  subtitle: {
    margin: 0,
    fontSize: '14px',
    color: '#4B5563',
  },
  configBar: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-end',
    backgroundColor: '#F9FAFB',
    border: '1px solid #E5E7EB',
    borderRadius: '8px',
    padding: '16px',
    marginBottom: '20px',
  },
  inputGroup: {
    flex: 2,
  },
  inputGroupSmall: {
    flex: 1,
  },
  label: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    fontSize: '13px',
    fontWeight: '600',
    color: '#374151',
  },
  input: {
    padding: '8px 12px',
    fontSize: '14px',
    border: '1px solid #D1D5DB',
    borderRadius: '6px',
  },
  inputSmall: {
    padding: '8px 12px',
    fontSize: '14px',
    border: '1px solid #D1D5DB',
    borderRadius: '6px',
  },
  actionGroup: {
    display: 'flex',
  },
  btnStart: {
    backgroundColor: '#2563EB',
    color: '#FFFFFF',
    border: 'none',
    padding: '9px 20px',
    fontSize: '14px',
    fontWeight: '600',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  btnStop: {
    backgroundColor: '#DC2626',
    color: '#FFFFFF',
    border: 'none',
    padding: '9px 20px',
    fontSize: '14px',
    fontWeight: '600',
    borderRadius: '6px',
    cursor: 'pointer',
  },
  errorBanner: {
    backgroundColor: '#FEE2E2',
    color: '#B91C1C',
    border: '1px solid #F87171',
    borderRadius: '6px',
    padding: '12px 16px',
    marginBottom: '20px',
    fontSize: '14px',
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr',
    gap: '24px',
  },
  videoCard: {
    backgroundColor: '#000000',
    borderRadius: '8px',
    overflow: 'hidden',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
  },
  videoContainer: {
    position: 'relative',
    width: '100%',
    aspectRatio: '16/9',
    backgroundColor: '#111827',
  },
  video: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    display: 'block',
  },
  videoPlaceholder: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#9CA3AF',
  },
  telemetryBar: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '12px 16px',
    backgroundColor: '#1F2937',
    color: '#F9FAFB',
    fontSize: '12px',
  },
  code: {
    backgroundColor: '#374151',
    padding: '2px 6px',
    borderRadius: '4px',
    fontFamily: 'monospace',
  },
  panelCard: {
    backgroundColor: '#FFFFFF',
    border: '1px solid #E5E7EB',
    borderRadius: '8px',
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
  },
  panelTitle: {
    margin: '0 0 12px 0',
    fontSize: '16px',
    fontWeight: '600',
  },
  statsSummary: {
    display: 'flex',
    gap: '8px',
    marginBottom: '16px',
  },
  statBadgeGreen: {
    backgroundColor: '#D1FAE5',
    color: '#065F46',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: '600',
  },
  statBadgeYellow: {
    backgroundColor: '#FEF3C7',
    color: '#92400E',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: '600',
  },
  tracksList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    overflowY: 'auto',
    maxHeight: '480px',
  },
  emptyText: {
    color: '#9CA3AF',
    fontSize: '14px',
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: '40px',
  },
  trackCard: {
    border: '1px solid #E5E7EB',
    borderRadius: '6px',
    padding: '10px',
    backgroundColor: '#F9FAFB',
  },
  trackCardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px',
  },
  trackDetails: {
    fontSize: '12px',
    color: '#6B7280',
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
};

