import { useRef, useEffect, useCallback } from 'react';
import { UI } from '../state/actions.js';

const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY = 3000; // 3 seconds
const MAX_RECONNECT_DELAY = 30000; // 30 seconds

/**
 * Manages a WebSocket connection to the game server.
 * Automatically reconnects with exponential backoff on disconnection.
 *
 * @param {Function} dispatch - React useReducer dispatch
 * @param {string|null} playerId - Current player's ID
 * @param {string|null} playerName - Current player's display name
 * @param {string} playerType - Player type: "HUMAN" or "HOST"
 * @returns {{ send: Function, connected: boolean }}
 */
export function useWebSocket(dispatch, playerId, playerName, playerType = 'HUMAN') {
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const connectedRef = useRef(false);

  const connect = useCallback(() => {
    if (!playerId || !playerName) return;

    // Build the WebSocket URL from the current page host
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.host;
    const url = `${protocol}://${host}/ws/${playerId}?name=${encodeURIComponent(playerName)}&type=${playerType}`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      connectedRef.current = true;
      reconnectAttemptRef.current = 0;
      dispatch({ type: UI.SET_CONNECTED, data: true });
      dispatch({ type: UI.ADD_TOAST, data: 'Connected' });
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        dispatch({ type: msg.event, data: msg.data });
      } catch (err) {
        console.error('[WS] Failed to parse message:', err);
      }
    };

    ws.onclose = () => {
      connectedRef.current = false;
      dispatch({ type: UI.SET_CONNECTED, data: false });

      // Attempt reconnection with exponential backoff
      if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        const attempt = reconnectAttemptRef.current;
        const delay = Math.min(
          BASE_RECONNECT_DELAY * Math.pow(2, attempt),
          MAX_RECONNECT_DELAY,
        );
        reconnectAttemptRef.current = attempt + 1;
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };
  }, [playerId, playerName, playerType, dispatch]);

  // Connect / reconnect when playerId or playerName change
  useEffect(() => {
    if (!playerId || !playerName) return;

    connect();

    return () => {
      // Cleanup: close socket and cancel any pending reconnect
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on intentional close
        wsRef.current.close();
        wsRef.current = null;
      }
      reconnectAttemptRef.current = 0;
    };
  }, [connect]);

  const send = useCallback(
    (data) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(data));
      } else {
        dispatch({ type: UI.ADD_TOAST, data: 'Not connected' });
      }
    },
    [dispatch],
  );

  return { send, connected: connectedRef.current };
}

export default useWebSocket;
