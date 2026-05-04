import { useState } from 'react';
import { getPlayerName } from '../utils';

const FORM_VIEW_URL =
  'https://docs.google.com/forms/d/e/1FAIpQLScB8cVpgvpKpgnwc26nmGNmatXm4ym1KIoN932t8LpPJbPAnA/viewform?usp=header';
const FORM_EMBED_URL =
  'https://docs.google.com/forms/d/e/1FAIpQLScB8cVpgvpKpgnwc26nmGNmatXm4ym1KIoN932t8LpPJbPAnA/viewform?embedded=true';

export default function GameOver({ finalScores, players, playerId, sessionId }) {
  const scores = finalScores?.final_scores || {};
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a);

  const [copied, setCopied] = useState(false);
  const [formOpen, setFormOpen] = useState(true);

  function copyParticipantId() {
    if (!playerId) return;
    navigator.clipboard.writeText(playerId).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div className="game-over-screen">
      <div className="game-over-container">
        <h1 className="game-over-title">Game Over</h1>
        <p>Thanks for playing The Sandbox!</p>

        {sorted.length > 0 && (
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(([pid, score], i) => {
                const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
                return (
                  <tr key={pid} className={rankClass || undefined}>
                    <td>{i + 1}</td>
                    <td>{getPlayerName(players, pid)}</td>
                    <td>{score}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <div className="research-form-section">
          <div className="research-form-header">
            <h2>One last thing — research feedback</h2>
            <p className="research-form-blurb">
              Please fill out this short post-game form so we can use your data. Your responses
              will be linked to this session's log via the Participant ID below.
            </p>
          </div>

          <div className="participant-id-card">
            <div className="participant-id-label">Your Participant ID</div>
            <div className="participant-id-value">
              <code>{playerId || 'unknown'}</code>
              <button
                type="button"
                className="btn-copy-id"
                onClick={copyParticipantId}
                disabled={!playerId}
              >
                {copied ? 'Copied!' : 'Copy'}
              </button>
            </div>
            {sessionId && (
              <div className="participant-session-line">
                Session: <code>{sessionId}</code>
              </div>
            )}
            <p className="participant-id-help">
              Paste this into the <strong>Participant ID</strong> field in the form below.
            </p>
          </div>

          <div className="research-form-actions">
            <button
              type="button"
              className="btn-form-toggle"
              onClick={() => setFormOpen((v) => !v)}
            >
              {formOpen ? 'Hide form' : 'Show form'}
            </button>
            <a
              className="btn-form-newtab"
              href={FORM_VIEW_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open form in new tab ↗
            </a>
          </div>

          {formOpen && (
            <div className="research-form-frame">
              <iframe
                src={FORM_EMBED_URL}
                title="Post-game research form"
                width="100%"
                height="900"
                frameBorder="0"
                marginHeight="0"
                marginWidth="0"
              >
                Loading form…
              </iframe>
            </div>
          )}
        </div>

        <button className="btn-play-again" onClick={() => window.location.reload()}>
          Play Again
        </button>
      </div>
    </div>
  );
}
