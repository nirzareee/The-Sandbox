import { useState } from 'react';
import { getPlayerName, getAvatarColor, getInitials } from '../utils';
import { UI } from '../state/actions';

export default function VotingPanel({ candidates, playerId, hasVoted, players, send, dispatch }) {
  const [votedFor, setVotedFor] = useState(null);

  const otherCandidates = candidates.filter((pid) => pid !== playerId);

  function handleVote(targetPid) {
    send({ action: 'VOTE', target_player: targetPid });
    dispatch({ type: UI.SET_VOTED });
    setVotedFor(targetPid);
  }

  return (
    <div className="voting-panel">
      <h2>Vote for the Imposter</h2>
      <p className="vote-status">
        {hasVoted ? `You voted for ${getPlayerName(players, votedFor)}.` : 'Select who you think is the imposter.'}
      </p>
      <div className="voting-grid">
        {otherCandidates.map((pid) => (
          <div key={pid} className={`voting-card${votedFor === pid ? ' voted' : ''}`}>
            <div
              className="player-avatar"
              style={{ backgroundColor: getAvatarColor(pid) }}
            >
              {getInitials(getPlayerName(players, pid))}
            </div>
            <span className="player-name">{getPlayerName(players, pid)}</span>
            <button
              className={`vote-btn${votedFor === pid ? ' selected' : ''}`}
              onClick={() => handleVote(pid)}
              disabled={hasVoted}
            >
              {votedFor === pid ? 'Voted' : 'Vote'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
