import TimerDisplay from './TimerDisplay';
import { getAvatarColor, getInitials, getPlayerName } from '../utils';

const PHASE_NAMES = {
  HINT: 'Hint Phase',
  DISCUSSION: 'Discussion',
  VOTING: 'Voting',
  RESULTS: 'Results',
  REFLECTION: 'Reflection',
  ASSIGNMENT: 'Round Starting',
};

// Short, player-friendly description of what the role means right now.
const ROLE_SUBTITLES = {
  INSIDER: 'You know the word — find the imposter',
  IMPOSTER: "Blend in — you don't know the word",
};

export default function GameHeader({
  round,
  numRounds,
  phase,
  timerSeconds,
  timerActive,
  role,
  topic,
  secretWord,
  playerId,
  playerName,
  players,
}) {
  const phaseName = PHASE_NAMES[phase] || phase;
  const displayName = playerName ?? getPlayerName(players ?? {}, playerId);
  const avatarColor = playerId ? getAvatarColor(playerId) : '#888';
  const initials = getInitials(displayName ?? '?');
  const roleLower = (role ?? '').toLowerCase();
  const showSecretArea = role === 'INSIDER' || role === 'IMPOSTER';

  return (
    <div className="game-header">
      {/* LEFT — round/phase + player identity */}
      <div className="game-header-left">
        <div className="header-round-block">
          <span className="round-indicator">Round {round}/{numRounds}</span>
          <span className="phase-indicator">{phaseName}</span>
        </div>
        {playerId && (
          <div className="you-are-card" title={`You are ${displayName}`}>
            <div
              className="player-avatar player-avatar-sm you-are-avatar"
              style={{ backgroundColor: avatarColor }}
            >
              {initials}
            </div>
            <div className="you-are-text">
              <span className="you-are-label">YOU</span>
              <span className="you-are-name">{displayName}</span>
            </div>
          </div>
        )}
      </div>

      {/* CENTER — topic + secret word */}
      <div className="game-header-center">
        {topic && (
          <div className="topic-chip">
            <span className="topic-chip-label">Topic</span>
            <span className="topic-chip-value">{topic}</span>
          </div>
        )}
        {showSecretArea && (
          <div className={`secret-card secret-card-${roleLower}`}>
            <span className="secret-card-label">
              {role === 'IMPOSTER' ? 'Secret Word' : 'Your Secret Word'}
            </span>
            <span className="secret-card-value">
              {role === 'IMPOSTER' ? '???' : (secretWord ?? '—')}
            </span>
          </div>
        )}
      </div>

      {/* RIGHT — role card + timer */}
      <div className="game-header-right">
        {role && (
          <div className={`role-card role-card-${roleLower}`} data-role={role}>
            <div className="role-card-badge">{role}</div>
            <div className="role-card-subtitle">{ROLE_SUBTITLES[role] ?? ''}</div>
          </div>
        )}
        <TimerDisplay seconds={timerSeconds} active={timerActive} />
      </div>
    </div>
  );
}
