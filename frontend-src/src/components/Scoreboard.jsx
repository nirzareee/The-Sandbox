import { getPlayerName, getAvatarColor, getInitials } from '../utils';

export default function Scoreboard({ players, cumulativeScores, playerId }) {
  const playerIds = Object.keys(players);

  const sorted = [...playerIds].sort(
    (a, b) => (cumulativeScores[b] ?? 0) - (cumulativeScores[a] ?? 0)
  );

  return (
    <>
      <h2>Scoreboard</h2>
      {sorted.map((pid) => (
        <div key={pid} className={`player-card${pid === playerId ? ' current' : ''}`}>
          <div className="player-card-info">
            <div
              className="player-avatar player-avatar-sm"
              style={{ backgroundColor: getAvatarColor(pid) }}
            >
              {getInitials(getPlayerName(players, pid))}
            </div>
            <span className="player-name">{getPlayerName(players, pid)}</span>
          </div>
          <span className="player-score">{cumulativeScores[pid] ?? 0}</span>
        </div>
      ))}
    </>
  );
}
