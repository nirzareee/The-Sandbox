import { useReducer } from 'react';
import { gameReducer } from './state/reducer';
import { initialState } from './state/initialState';
import { useWebSocket } from './hooks/useWebSocket';
import { useTimer } from './hooks/useTimer';
import ConnectionStatus from './components/ConnectionStatus';
import Toast from './components/Toast';
import Lobby from './components/Lobby';
import GameScreen from './components/GameScreen';
import MetaGame from './components/MetaGame';
import Survey from './components/Survey';
import GrandReveal from './components/GrandReveal';
import GameOver from './components/GameOver';
import HostDashboard from './components/HostDashboard';

const GAME_PHASES = ['ASSIGNMENT', 'HINT', 'DISCUSSION', 'VOTING', 'RESULTS', 'REFLECTION'];

export default function App() {
  const [state, dispatch] = useReducer(gameReducer, initialState);
  const playerType = state.isHost ? 'HOST' : 'HUMAN';
  const { send, connected } = useWebSocket(dispatch, state.playerId, state.playerName, playerType);
  useTimer(dispatch, state.timerActive, state.timerSeconds);

  return (
    <>
      <ConnectionStatus connected={connected} />
      <Toast toasts={state.toasts} dispatch={dispatch} />

      {state.isHost ? (
        <HostDashboard state={state} dispatch={dispatch} send={send} />
      ) : (
        <>
          {(!state.phase || state.phase === 'LOBBY') && (
            <Lobby players={state.players} dispatch={dispatch} connected={connected} />
          )}

          {GAME_PHASES.includes(state.phase) && (
            <GameScreen state={state} dispatch={dispatch} send={send} />
          )}

          {state.phase === 'META_GAME' && (
            <MetaGame
              metaGamePlayers={state.metaGamePlayers}
              playerId={state.playerId}
              send={send}
              timerSeconds={state.timerSeconds}
              timerActive={state.timerActive}
              classificationSubmitted={state.classificationSubmitted}
            />
          )}

          {state.phase === 'SURVEY' && (
            <Survey
              send={send}
              timerSeconds={state.timerSeconds}
              timerActive={state.timerActive}
              surveySubmitted={state.surveySubmitted}
            />
          )}

          {state.phase === 'REVEAL' && (
            <GrandReveal revealData={state.revealData} players={state.players} send={send} />
          )}

          {state.phase === 'FINISHED' && (
            <GameOver finalScores={state.finalScores} players={state.players} />
          )}
        </>
      )}
    </>
  );
}
