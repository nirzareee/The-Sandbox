import TimerDisplay from './TimerDisplay';

const PHASE_NAMES = {
  HINT: 'Hint Phase',
  DISCUSSION: 'Discussion',
  VOTING: 'Voting',
  RESULTS: 'Results',
  REFLECTION: 'Reflection',
  ASSIGNMENT: 'Round Starting',
};

export default function GameHeader({ round, numRounds, phase, timerSeconds, timerActive, role, topic, secretWord }) {
  const phaseName = PHASE_NAMES[phase] || phase;

  return (
    <div className="game-header">
      <div className="game-header-left">
        <span className="round-indicator">Round {round}/{numRounds}</span>
        <span className="phase-indicator">{phaseName}</span>
      </div>
      <div className="game-header-center">
        {topic && <span className="topic-display">{topic}</span>}
        {secretWord && (
          <span className="secret-word-display">
            {role === 'IMPOSTER' ? '???' : secretWord}
          </span>
        )}
      </div>
      <div className="game-header-right">
        {role && <span className="role-badge" data-role={role}>{role}</span>}
        <TimerDisplay seconds={timerSeconds} active={timerActive} />
      </div>
    </div>
  );
}
