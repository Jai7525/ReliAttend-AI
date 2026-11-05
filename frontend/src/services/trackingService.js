/**
 * ReliAttend AI — Frontend Tracking API Client (Phase 9 - Substage 9.4)
 * Handles authenticated API communication with FastAPI tracking endpoints.
 * Privacy by design: no raw frames or biometrics stored in client storage.
 */

const API_BASE_URL = '/api/v1';

export class TrackingApiClient {
  /**
   * @param {string} token - Bearer JWT token
   */
  constructor(token = '') {
    this.token = token;
  }

  setToken(token) {
    this.token = token;
  }

  _getHeaders(includeContentType = true) {
    const headers = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    if (includeContentType) {
      headers['Content-Type'] = 'application/json';
    }
    return headers;
  }

  /**
   * Create an in-memory tracking session
   * POST /api/v1/tracking/sessions
   */
  async createSession(classSectionId, config = {}) {
    const response = await fetch(`${API_BASE_URL}/tracking/sessions`, {
      method: 'POST',
      headers: this._getHeaders(true),
      body: JSON.stringify({
        class_section_id: Number(classSectionId),
        ...config,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail?.message || err.message || `Failed to create session (${response.status})`);
    }

    return await response.json();
  }

  /**
   * Process a single sequential image frame
   * POST /api/v1/tracking/sessions/{session_id}/process-frame
   */
  async processFrame(sessionId, frameSeq, timestamp, imageBlob, timeoutMs = 8000) {
    const formData = new FormData();
    formData.append('frame_seq', frameSeq);
    if (timestamp !== undefined && timestamp !== null) {
      formData.append('timestamp', timestamp);
    }
    formData.append('file', imageBlob, `frame_${frameSeq}.jpg`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${API_BASE_URL}/tracking/sessions/${sessionId}/process-frame`, {
        method: 'POST',
        headers: this._getHeaders(false), // FormData handles multipart boundary automatically
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail?.message || err.message || `Frame processing failed (${response.status})`);
      }

      return await response.json();
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error(`Frame processing timed out after ${timeoutMs}ms`);
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Query active tracks
   * GET /api/v1/tracking/sessions/{session_id}/tracks
   */
  async getTracks(sessionId) {
    const response = await fetch(`${API_BASE_URL}/tracking/sessions/${sessionId}/tracks`, {
      method: 'GET',
      headers: this._getHeaders(true),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail?.message || err.message || `Failed to fetch tracks (${response.status})`);
    }

    return await response.json();
  }

  /**
   * Explicitly terminate tracking session
   * DELETE /api/v1/tracking/sessions/{session_id}
   */
  async endSession(sessionId) {
    if (!sessionId) return null;
    const response = await fetch(`${API_BASE_URL}/tracking/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: this._getHeaders(true),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail?.message || err.message || `Failed to terminate session (${response.status})`);
    }

    return await response.json();
  }
}

export const trackingClient = new TrackingApiClient();

