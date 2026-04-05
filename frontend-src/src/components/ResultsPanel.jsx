import { getPlayerName } from '../utils';

export default function ResultsPanel({ imposterId, imposterCaught, votes, correctVoters, roundScores, cumulativeScores, players }) {
  const imposterName = getPlayerName(players, imposterId);
  const playerIds = Object.keys(players);

  // Compute vote tallies: how many votes each target received
  const tallies = {};
  for (const targetId of Object.values(votes)) {
    tallies[targetId] = (tallies[targetId] || 0) + 1;
  }
  // Sort targets by vote count descending
  const sortedTargets = Object.entries(tallies).sort((a, b) => b[1] - a[1]);

  return (
    <div className="results-panel">
      <div className="imposter-reveal">
        <span className="results-imposter-name">The imposter was...</span>
        <span className="imposter-name-highlight">{imposterName}</span>
        <span className={imposterCaught ? 'imposter-caught' : 'imposter-escaped'}>
          {imposterCaught ? 'CAUGHT!' : 'ESCAPED!'}
        </span>
      </div>

      <h3>Votes</h3>
      {sortedTargets.length > 0 && (
        <div className="vote-tallies">
          {sortedTargets.map(([targetId, count]) => (
            <div key={targetId} className={`vote-tally-row ${targetId === imposterId ? 'vote-tally-imposter' : ''}`}>
              <span className="vote-tally-target">{getPlayerName(players, targetId)}</span>
              <span className="vote-tally-bar-wrap">
                <span
                  className="vote-tally-bar"
                  style={{ width: `${(count / (Object.keys(votes).length || 1)) * 100}%` }}
                />
              </span>
              <span className="vote-tally-count">{count} vote{count !== 1 ? 's' : ''}</span>
            </div>
          ))}
        </div>
      )}

      <div className="vote-detail-table">
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Voted For</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {playerIds.map((pid) => {
              const targetId = votes[pid];
              const isCorrect = correctVoters.includes(pid);
              return (
                <tr key={pid} className={isCorrect ? 'vote-correct' : ''}>
                  <td>{getPlayerName(players, pid)}{pid === imposterId ? ' (imposter)' : ''}</td>
                  <td>{targetId ? getPlayerName(players, targetId) : <span className="vote-no-vote">didn't vote</span>}</td>
                  <td className="vote-result-icon">{isCorrect ? '\u2714' : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>Scores</h3>
      <table className="scores-table">
        <thead>
          <tr>
            <th>Player</th>
            <th>Round</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          {playerIds.map((pid) => (
            <tr key={pid}>
              <td>{getPlayerName(players, pid)}</td>
              <td>{roundScores[pid] ?? 0}</td>
              <td>{cumulativeScores[pid] ?? 0}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
