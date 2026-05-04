import { useState } from 'react';
import { getPlayerName } from '../utils';

export default function MetaGame({ metaGamePlayers, playerId, send, timerSeconds, timerActive, classificationSubmitted }) {
  const otherPlayers = Object.keys(metaGamePlayers).filter(pid => pid !== playerId);

  const [classifications, setClassifications] = useState(() => {
    const init = {};
    for (const pid of otherPlayers) {
      init[pid] = { guess: '', confidence: 3, reasoning: '' };
    }
    return init;
  });
  const [submitted, setSubmitted] = useState(false);

  function updateField(pid, field, value) {
    setClassifications(prev => ({
      ...prev,
      [pid]: { ...prev[pid], [field]: value },
    }));
  }

  function handleSubmit() {
    const allClassified = otherPlayers.every(pid => classifications[pid].guess);
    if (!allClassified) return;

    const payload = {};
    for (const pid of otherPlayers) {
      payload[pid] = {
        guess: classifications[pid].guess,
        confidence: parseInt(classifications[pid].confidence, 10),
        reasoning: classifications[pid].reasoning,
      };
    }

    send({ action: 'CLASSIFICATION', content: JSON.stringify(payload) });
    setSubmitted(true);
  }

  const isDisabled = submitted || classificationSubmitted;

  return (
    <div className="meta-game-screen">
      <div className="meta-game-content">
        <h1>Who's Human? Who's AI?</h1>
        <p className="meta-game-instructions">
          Classify each player as Human or AI. Rate your confidence and explain your reasoning.
        </p>

        {isDisabled && (
          <div className="meta-game-waiting-notice">
            Classifications submitted. Please wait until every player has submitted &mdash;
            the reveal will appear automatically.
          </div>
        )}

        <div className="classification-form">
          {otherPlayers.map(pid => (
            <div key={pid} className="classification-card">
              <span className="classification-player-name">
                {getPlayerName(metaGamePlayers, pid)}
              </span>

              <div className="toggle-switch">
                <label className="classification-radio-label">
                  <input
                    type="radio"
                    name={`guess-${pid}`}
                    value="human"
                    checked={classifications[pid]?.guess === 'human'}
                    onChange={() => updateField(pid, 'guess', 'human')}
                    disabled={isDisabled}
                  />
                  <span>Human</span>
                </label>
                <label className="classification-radio-label">
                  <input
                    type="radio"
                    name={`guess-${pid}`}
                    value="ai"
                    checked={classifications[pid]?.guess === 'ai'}
                    onChange={() => updateField(pid, 'guess', 'ai')}
                    disabled={isDisabled}
                  />
                  <span>AI</span>
                </label>
              </div>

              <div className="classification-confidence">
                <label>
                  Confidence: {classifications[pid]?.confidence}
                  <input
                    type="range"
                    className="confidence-slider"
                    min={1}
                    max={5}
                    value={classifications[pid]?.confidence}
                    onChange={e => updateField(pid, 'confidence', e.target.value)}
                    disabled={isDisabled}
                  />
                </label>
                <div className="confidence-labels">
                  <span>Wild guess</span>
                  <span>Somewhat sure</span>
                  <span>Certain</span>
                </div>
              </div>

              <textarea
                className="classification-reasoning"
                placeholder="What made you think this player is human or AI?"
                value={classifications[pid]?.reasoning || ''}
                onChange={e => updateField(pid, 'reasoning', e.target.value)}
                disabled={isDisabled}
              />
            </div>
          ))}
        </div>

        <div className="meta-game-footer">
          <button
            onClick={handleSubmit}
            disabled={isDisabled || !otherPlayers.every(pid => classifications[pid]?.guess)}
          >
            Submit Classifications
          </button>
        </div>
      </div>
    </div>
  );
}
