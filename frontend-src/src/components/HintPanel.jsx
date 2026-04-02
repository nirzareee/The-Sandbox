import { useState, useEffect } from 'react';
import { getPlayerName, getAvatarColor, getInitials } from '../utils';

function countWords(text) {
  return text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
}

export default function HintPanel({ hints, hintSubmitted, hintsExpected, allHintsIn, readingCountdown, players, playerId, hintMaxChars = 60, hintMaxWords = 8, send }) {
  const [text, setText] = useState('');
  const [submitted, setSubmitted] = useState(false);

  // Reset local submitted state when round changes (hintSubmitted resets)
  useEffect(() => {
    if (!hintSubmitted) setSubmitted(false);
  }, [hintSubmitted]);

  const charCount = text.length;
  const wordCount = countWords(text);
  const overLimit = charCount > hintMaxChars || wordCount > hintMaxWords;
  const hintsCount = Object.keys(hints).length;
  const canSubmit = !submitted && !hintSubmitted;

  function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed || trimmed.length > hintMaxChars || countWords(trimmed) > hintMaxWords) return;
    send({ action: 'HINT', content: trimmed });
    setSubmitted(true);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="hint-panel">
      <div className="hint-progress">
        <h2>Hints</h2>
        <span className="hint-count-pill">{hintsCount}/{hintsExpected} submitted</span>
      </div>

      {/* Submitted hints */}
      <div className="submitted-hints">
        {Object.entries(hints).map(([pid, hint]) => (
          <div key={pid} className="hint-card">
            <div
              className="player-avatar player-avatar-sm"
              style={{ backgroundColor: getAvatarColor(pid) }}
            >
              {getInitials(getPlayerName(players, pid))}
            </div>
            <div className="hint-card-body">
              <span className="hint-card-name">{getPlayerName(players, pid)}</span>
              <span className="hint-card-text">{hint}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Reading period notice */}
      {allHintsIn && (
        <div className="reading-period-notice">
          All hints are in! Reading period: {readingCountdown}s
        </div>
      )}

      {/* Input area — always shown, disabled after submission */}
      {canSubmit && !allHintsIn && (
        <div className="hint-input-area">
          <input
            id="hint-input"
            type="text"
            maxLength={hintMaxChars}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter your hint..."
          />
          <span className={`char-counter${overLimit ? ' counter-over' : ''}`}>
            {charCount}/{hintMaxChars} chars, {wordCount}/{hintMaxWords} words
          </span>
          <button className="hint-submit-btn" onClick={handleSubmit} disabled={overLimit || !text.trim()}>
            Submit Hint
          </button>
        </div>
      )}

      {(submitted || hintSubmitted) && !allHintsIn && (
        <p className="status-message">Hint submitted! Waiting for others...</p>
      )}
    </div>
  );
}
