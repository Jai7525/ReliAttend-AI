import { useEffect, useRef } from 'react';

/**
 * Renders spatial tracking bounding boxes and state tags strictly aligned
 * with the underlying video stream, taking aspect ratio scaling and letterboxing into account.
 */
export function VideoOverlayCanvas({ videoRef, tracks = [] }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas buffer size to client display dimensions
    const clientWidth = video.clientWidth;
    const clientHeight = video.clientHeight;

    if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
      canvas.width = clientWidth;
      canvas.height = clientHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const videoNativeWidth = video.videoWidth;
    const videoNativeHeight = video.videoHeight;

    if (!videoNativeWidth || !videoNativeHeight || !tracks || tracks.length === 0) {
      return;
    }

    // Compute letterbox/pillarbox offset and uniform scaling factor (object-fit: contain emulation)
    const videoRatio = videoNativeWidth / videoNativeHeight;
    const canvasRatio = clientWidth / clientHeight;

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;

    if (canvasRatio > videoRatio) {
      // Pillarboxed (black bars on left/right)
      scale = clientHeight / videoNativeHeight;
      offsetX = (clientWidth - videoNativeWidth * scale) / 2;
    } else {
      // Letterboxed (black bars on top/bottom)
      scale = clientWidth / videoNativeWidth;
      offsetY = (clientHeight - videoNativeHeight * scale) / 2;
    }

    // Draw each track box and label
    tracks.forEach((track) => {
      const bbox = track.bounding_box;
      if (!bbox) return;

      const screenX = offsetX + bbox.x * scale;
      const screenY = offsetY + bbox.y * scale;
      const screenW = bbox.width * scale;
      const screenH = bbox.height * scale;

      // Color coding based on lifecycle state & recognition status
      let strokeColor = '#10B981'; // Green: confirmed
      let labelBg = '#059669';

      if (track.is_ambiguous) {
        strokeColor = '#EF4444'; // Red: ambiguous crossing
        labelBg = '#DC2626';
      } else if (track.recognized_student?.identity_status === 'conflict') {
        strokeColor = '#DC2626'; // Red: identity conflict
        labelBg = '#B91C1C';
      } else if (track.recognized_student?.identity_status === 'assigned') {
        strokeColor = '#2563EB'; // Blue: recognized & assigned
        labelBg = '#1D4ED8';
      } else if (track.state === 'tentative') {
        strokeColor = '#FBBF24'; // Yellow: tentative
        labelBg = '#D97706';
      } else if (track.state === 'temporarily_lost') {
        strokeColor = '#F97316'; // Orange: lost / extrapolated
        labelBg = '#EA580C';
      }

      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;

      if (track.state === 'temporarily_lost') {
        ctx.setLineDash([6, 6]);
      } else {
        ctx.setLineDash([]);
      }

      ctx.strokeRect(screenX, screenY, screenW, screenH);
      ctx.setLineDash([]);

      // Label badge: Display Track ID and masked student identifier by default (no similarity percentage)
      let labelText = `${track.track_id} [${track.state.toUpperCase()}]`;
      if (track.is_ambiguous) {
        labelText += ' [AMBIGUOUS]';
      } else if (track.recognized_student?.identity_status === 'conflict') {
        labelText += ' [CONFLICT]';
      } else if (track.recognized_student?.identity_status === 'assigned') {
        labelText = `${track.track_id} | ${track.recognized_student.student_id_masked}`;
      }
      ctx.font = 'bold 12px sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const badgeWidth = textMetrics.width + 10;
      const badgeHeight = 20;

      const badgeY = Math.max(0, screenY - badgeHeight - 4);
      ctx.fillStyle = labelBg;
      ctx.fillRect(screenX, badgeY, badgeWidth, badgeHeight);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(labelText, screenX + 5, badgeY + 14);
    });
  }, [videoRef, tracks]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}

