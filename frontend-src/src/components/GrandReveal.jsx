import { useState, useMemo } from 'react';
import { getPlayerName } from '../utils';

export default function GrandReveal({ revealData, players, send }) {
  const [step, setStep] = useState(0);

  if (!revealData) return null;

  const { identities, leaderboard, classification_scores, confusion_matrix, all_classifications } = revealData;

  const playerEntries = useMemo(
    () => (identities ? Object.entries(identities) : []),
    [identities]
  );

  // Steps: 0..N-1 = one player each, N = leaderboard, N+1 = stats
  const totalSteps = playerEntries.length + 2;
  const isPlayerStep = step < playerEntries.length;
  const isLeaderboardStep = step === playerEntries.length;
  const isStatsStep = step === playerEntries.length + 1;
  const isLast = step >= totalSteps - 1;

  const getName = (pid) => identities?.[pid]?.display_name || getPlayerName(players, pid);
  const getType = (pid) => identities?.[pid]?.type;

  const allTargetPids = identities ? Object.keys(identities) : [];
  const allVoterPids = all_classifications ? Object.keys(all_classifications) : [];

  function handleNext() {
    if (isLast) {
      send({ action: 'ADVANCE_PHASE', content: '' });
    } else {
      setStep(s => s + 1);
    }
  }

  return (
    <div className="grand-reveal-screen">
      <div className="grand-reveal-content">
        <h1>The Grand Reveal</h1>

        {/* Previously revealed cards (small) */}
        {isPlayerStep && step > 0 && (
          <div className="identity-cards identity-cards-small">
            {playerEntries.slice(0, step).map(([pid, info]) => (
              <div
                key={pid}
                className="identity-card identity-card-small"
                data-player-type={info.type}
              >
                <h3>{info.display_name}</h3>
                <span className={`type-badge ${info.type === 'HUMAN' ? 'human' : 'ai'}`}>
                  {info.type === 'HUMAN' ? 'Human' : `AI`}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Current player reveal */}
        {isPlayerStep && (() => {
          const [pid, info] = playerEntries[step];
          return (
            <div className="identity-cards">
              <div
                className="identity-card identity-card-reveal"
                data-player-type={info.type}
              >
                <h3>{info.display_name}</h3>
                <span className={`type-badge ${info.type === 'HUMAN' ? 'human' : 'ai'}`}>
                  {info.type === 'HUMAN' ? 'Human' : `AI - ${info.model_name || 'Unknown'}`}
                </span>
              </div>
            </div>
          );
        })()}

        {/* Leaderboard step */}
        {isLeaderboardStep && (
          <>
            {/* All identity cards recap */}
            <div className="identity-cards identity-cards-small">
              {playerEntries.map(([pid, info]) => (
                <div
                  key={pid}
                  className="identity-card identity-card-small"
                  data-player-type={info.type}
                >
                  <h3>{info.display_name}</h3>
                  <span className={`type-badge ${info.type === 'HUMAN' ? 'human' : 'ai'}`}>
                    {info.type === 'HUMAN' ? 'Human' : 'AI'}
                  </span>
                </div>
              ))}
            </div>

            {leaderboard && leaderboard.length > 0 && (
              <div className="reveal-leaderboard">
                <h2>Leaderboard</h2>
                <table className="leaderboard-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Player</th>
                      <th>Type</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((entry, i) => {
                      const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
                      const playerIdentity = identities?.[entry.player_id];
                      const playerType = playerIdentity?.type || '';
                      return (
                        <tr key={entry.player_id} className={rankClass || undefined}>
                          <td>{entry.rank ?? i + 1}</td>
                          <td>{playerIdentity?.display_name || getPlayerName(players, entry.player_id)}</td>
                          <td>{playerType}</td>
                          <td>{entry.score}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {/* Stats step — grid, details, confusion matrix */}
        {isStatsStep && (
          <>
            {all_classifications && allVoterPids.length > 0 && (
              <StatsSection
                allClassifications={all_classifications}
                allTargetPids={allTargetPids}
                allVoterPids={allVoterPids}
                getName={getName}
                getType={getType}
                players={players}
              />
            )}

            {classification_scores && Object.keys(classification_scores).length > 0 && (
              <div className="accuracy-summary">
                <h3>Classification Scores</h3>
                {Object.entries(classification_scores).map(([pid, score]) => (
                  <div key={pid} className="accuracy-entry">
                    <span>{getPlayerName(players, pid)}</span>
                    <span>{score}</span>
                  </div>
                ))}
              </div>
            )}

            {confusion_matrix?.aggregate && (
              <div className="confusion-matrix">
                <h3>Aggregate Accuracy</h3>
                <table>
                  <thead>
                    <tr>
                      <th></th>
                      <th>Predicted AI</th>
                      <th>Predicted Human</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Actual AI</th>
                      <td>{confusion_matrix.aggregate.tp}</td>
                      <td>{confusion_matrix.aggregate.fn}</td>
                    </tr>
                    <tr>
                      <th>Actual Human</th>
                      <td>{confusion_matrix.aggregate.fp}</td>
                      <td>{confusion_matrix.aggregate.tn}</td>
                    </tr>
                  </tbody>
                </table>
                <p className="status-message">
                  Overall accuracy: {(confusion_matrix.aggregate.accuracy * 100).toFixed(0)}%
                </p>
              </div>
            )}
          </>
        )}

        {/* Navigation */}
        <div className="reveal-nav">
          {step > 0 && (
            <button className="btn-reveal-nav" onClick={() => setStep(s => s - 1)}>
              Back
            </button>
          )}
          <span className="reveal-step-indicator">
            {step + 1} / {totalSteps}
          </span>
          <button className="btn-reveal-nav btn-reveal-primary" onClick={handleNext}>
            {isLast ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}

/* Extracted stats section to keep the main component readable */
function StatsSection({ allClassifications, allTargetPids, allVoterPids, getName, getType, players }) {
  const [expandedPlayer, setExpandedPlayer] = useState(null);

  return (
    <div className="meta-guesses-section">
      <h2>Who Guessed What</h2>

      <div className="meta-grid-wrap">
        <table className="meta-grid-table">
          <thead>
            <tr>
              <th className="meta-grid-corner"></th>
              {allTargetPids.map(tid => (
                <th key={tid} className="meta-grid-col-header">
                  <span className="meta-grid-name">{getName(tid)}</span>
                  <span className={`meta-grid-truth ${getType(tid) === 'HUMAN' ? 'human' : 'ai'}`}>
                    {getType(tid) === 'HUMAN' ? 'Human' : 'AI'}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {allVoterPids.map(vid => {
              const guesses = allClassifications[vid] || {};
              return (
                <tr key={vid}>
                  <td className="meta-grid-row-header">{getName(vid)}</td>
                  {allTargetPids.map(tid => {
                    if (tid === vid) {
                      return <td key={tid} className="meta-grid-self">&mdash;</td>;
                    }
                    const c = guesses[tid];
                    if (!c) {
                      return <td key={tid} className="meta-grid-empty"></td>;
                    }
                    const actual = getType(tid);
                    const isCorrect = actual && (
                      (c.guess === 'human' && actual === 'HUMAN') ||
                      (c.guess === 'ai' && actual === 'AI')
                    );
                    return (
                      <td key={tid} className={`meta-grid-cell ${isCorrect ? 'correct' : 'wrong'}`}>
                        <span className={`guess-badge ${c.guess}`}>{c.guess === 'human' ? 'H' : 'AI'}</span>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="meta-grid-legend">
        <span className="meta-legend-item"><span className="meta-legend-dot correct"></span> Correct</span>
        <span className="meta-legend-item"><span className="meta-legend-dot wrong"></span> Wrong</span>
        <span className="meta-legend-item"><span className="guess-badge human">H</span> = Human</span>
        <span className="meta-legend-item"><span className="guess-badge ai">AI</span> = AI</span>
      </div>

      <h3 style={{ marginTop: '1rem' }}>Details &amp; Reasoning</h3>
      {allVoterPids.map(pid => {
        const guesses = allClassifications[pid] || {};
        return (
          <div key={pid} className="player-guess-card">
            <button
              className={`player-guess-header ${expandedPlayer === pid ? 'expanded' : ''}`}
              onClick={() => setExpandedPlayer(expandedPlayer === pid ? null : pid)}
            >
              <span className="player-guess-name">{getName(pid)}</span>
              <span className="player-guess-toggle">{expandedPlayer === pid ? '\u25B2' : '\u25BC'}</span>
            </button>
            {expandedPlayer === pid && (
              <table className="player-guess-table">
                <thead>
                  <tr>
                    <th>Target</th>
                    <th>Guess</th>
                    <th>Confidence</th>
                    <th>Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(guesses).map(([targetPid, c]) => {
                    const actual = getType(targetPid);
                    const isCorrect = actual && (
                      (c.guess === 'human' && actual === 'HUMAN') ||
                      (c.guess === 'ai' && actual === 'AI')
                    );
                    return (
                      <tr key={targetPid} className={isCorrect ? 'guess-correct' : 'guess-wrong'}>
                        <td>{getName(targetPid)}</td>
                        <td>
                          <span className={`guess-badge ${c.guess}`}>
                            {c.guess.toUpperCase()}
                          </span>
                        </td>
                        <td>{'\u2605'.repeat(c.confidence)}{'\u2606'.repeat(5 - c.confidence)}</td>
                        <td className="reasoning-cell">{c.reasoning}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        );
      })}
    </div>
  );
}
