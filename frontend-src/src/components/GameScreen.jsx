import GameHeader from './GameHeader';
import ImageGrid from './ImageGrid';
import HintPanel from './HintPanel';
import ChatPanel from './ChatPanel';
import VotingPanel from './VotingPanel';
import ResultsPanel from './ResultsPanel';
import Scoreboard from './Scoreboard';

function ActivityPanel({ state, dispatch, send }) {
  const { phase } = state;

  switch (phase) {
    case 'HINT':
      return (
        <HintPanel
          hints={state.hints}
          hintSubmitted={state.hintSubmitted}
          hintsExpected={state.hintsExpected}
          allHintsIn={state.allHintsIn}
          readingCountdown={state.readingCountdown}
          players={state.players}
          playerId={state.playerId}
          hintMaxChars={state.hintMaxChars}
          hintMaxWords={state.hintMaxWords}
          send={send}
        />
      );

    case 'DISCUSSION':
    case 'REFLECTION':
      return (
        <ChatPanel
          players={state.players}
          playerId={state.playerId}
          chatMessages={state.chatMessages}
          dmMessages={state.dmMessages}
          unreadDMs={state.unreadDMs}
          activeTab={state.activeTab}
          dispatch={dispatch}
          send={send}
          phase={phase}
        />
      );

    case 'VOTING':
      return (
        <VotingPanel
          candidates={state.candidates}
          playerId={state.playerId}
          hasVoted={state.hasVoted}
          players={state.players}
          send={send}
          dispatch={dispatch}
        />
      );

    case 'RESULTS':
      return (
        <ResultsPanel
          imposterId={state.imposterId}
          imposterCaught={state.imposterCaught}
          votes={state.votes}
          correctVoters={state.correctVoters}
          roundScores={state.roundScores}
          cumulativeScores={state.cumulativeScores}
          players={state.players}
        />
      );

    case 'ASSIGNMENT':
      return (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
          <p className="status-message">Round starting...</p>
        </div>
      );

    default:
      return null;
  }
}

export default function GameScreen({ state, dispatch, send }) {
  return (
    <div className="game-screen">
      <GameHeader
        round={state.currentRound}
        numRounds={state.numRounds}
        phase={state.phase}
        timerSeconds={state.timerSeconds}
        timerActive={state.timerActive}
        role={state.role}
        topic={state.topic}
        secretWord={state.secretWord}
      />
      <div className="game-body">
        <div className="grid-area">
          <ImageGrid
            grid={state.grid}
            role={state.role}
            secretWord={state.secretWord}
            topic={state.topic}
          />
        </div>
        <div className="activity-area">
          <div key={state.phase} className="phase-content">
            <ActivityPanel state={state} dispatch={dispatch} send={send} />
          </div>
        </div>
        <div className="scoreboard-area">
          <Scoreboard
            players={state.players}
            cumulativeScores={state.cumulativeScores}
            playerId={state.playerId}
          />
        </div>
      </div>
    </div>
  );
}
