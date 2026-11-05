import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TrackingApiClient } from '../src/services/trackingService';

describe('TrackingApiClient', () => {
  let client;

  beforeEach(() => {
    client = new TrackingApiClient('test-jwt-token');
    global.fetch = vi.fn();
  });

  it('createSession sends correct headers and payload', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: 'trk_sess_123',
        class_section_id: 1,
        created_at: 1000.0,
      }),
    });

    const res = await client.createSession(1, { iou_threshold: 0.35 });

    expect(global.fetch).toHaveBeenCalledWith('/api/v1/tracking/sessions', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer test-jwt-token',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        class_section_id: 1,
        iou_threshold: 0.35,
      }),
    });
    expect(res.session_id).toBe('trk_sess_123');
  });

  it('processFrame sets strictly increasing frame_seq and form data', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: 'trk_sess_123',
        frame_seq: 0,
        tracks: [],
      }),
    });

    const mockBlob = new Blob(['fake_jpg'], { type: 'image/jpeg' });
    const res = await client.processFrame('trk_sess_123', 0, 100.5, mockBlob);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const callArgs = global.fetch.mock.calls[0];
    expect(callArgs[0]).toBe('/api/v1/tracking/sessions/trk_sess_123/process-frame');
    expect(callArgs[1].method).toBe('POST');
    expect(callArgs[1].headers).toEqual({
      Authorization: 'Bearer test-jwt-token',
    });
    expect(res.frame_seq).toBe(0);
  });

  it('endSession calls DELETE on session URL', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: 'trk_sess_123',
        message: 'Terminated',
      }),
    });

    const res = await client.endSession('trk_sess_123');
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/tracking/sessions/trk_sess_123', {
      method: 'DELETE',
      headers: {
        Authorization: 'Bearer test-jwt-token',
        'Content-Type': 'application/json',
      },
    });
    expect(res.message).toBe('Terminated');
  });

  it('throws descriptive error on HTTP failure', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      json: async () => ({
        detail: { message: 'Faculty member not authorized for class section' },
      }),
    });

    await expect(client.createSession(99)).rejects.toThrow(
      'Faculty member not authorized for class section'
    );
  });
});

