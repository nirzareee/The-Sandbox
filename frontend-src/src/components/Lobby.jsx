import { useState, useEffect } from 'react';
import { UI } from '../state/actions';
import { randomChars } from '../utils';

export default function Lobby({ players, dispatch, connected }) {
  const [playerName, setPlayerName] = useState('');
  const [joined, setJoined] = useState(false);

  const connectedCount = Object.values(players).filter(p => p.connected).length;

  // Auto-connect as host if ?host=1 is in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('host') === '1' && !joined) {
      const playerId = 'host_' + randomChars(4);
      dispatch({ type: UI.SET_PLAYER, data: { playerId, playerName: 'Host', isHost: true } });
      setJoined(true);
    }
  }, [dispatch, joined]);

  function handleJoin() {
    const trimmed = playerName.trim();
    if (!trimmed) return;
    const playerId = 'human_' + randomChars(4);
    dispatch({ type: UI.SET_PLAYER, data: { playerId, playerName: trimmed } });
    setJoined(true);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleJoin();
  }

  return (
    <div className="screen lobby-screen">
      <div className="lobby-content">
        <div className="lobby-branding">
          <h1 className="game-title">The Sandbox</h1>
          <p className="game-tagline">A Social Deduction Experiment</p>
        </div>

        <div className="lobby-join-section">
          <input
            id="name-input"
            type="text"
            maxLength={20}
            placeholder="Enter your name..."
            value={playerName}
            onChange={e => setPlayerName(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={joined}
          />
          <button className="btn-join" onClick={handleJoin} disabled={joined || !playerName.trim()}>
            Join Game
          </button>
        </div>

        <div className="lobby-players-section">
          <h2>Players</h2>
          <p className="status-message">
            {connectedCount} player{connectedCount !== 1 ? 's' : ''} connected
          </p>
          <p className="status-message" style={{ fontSize: '0.85rem', opacity: 0.7 }}>
            Names will be assigned when the game starts
          </p>
        </div>

        <p className="status-message" style={{ fontSize: '0.85rem', opacity: 0.7, marginTop: '16px' }}>
          Waiting for the host to start the game...
        </p>
      </div>
    </div>
  );
}
