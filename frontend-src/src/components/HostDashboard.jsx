import { useMemo, useState, useEffect, useCallback } from 'react';
import { UI } from '../state/actions.js';

const PHASE_LABELS = {
  LOBBY: 'LOBBY',
  ASSIGNMENT: 'ASSIGNMENT',
  HINT: 'HINTS',
  DISCUSSION: 'DISCUSSION',
  VOTING: 'VOTING',
  RESULTS: 'RESULTS',
  REFLECTION: 'REFLECTION',
  META_GAME: 'META-GAME',
  SURVEY: 'SURVEY',
  REVEAL: 'REVEAL',
  FINISHED: 'FINISHED',
};

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function getPlayerName(players, playerId) {
  return players[playerId]?.display_name ?? playerId;
}

// ---------------------------------------------------------------------------
// Activity badge (used in All Activity tab)
// ---------------------------------------------------------------------------
function ActivityBadge({ type }) {
  const styles = {
    group: { bg: '#4ecdc4', label: 'Group' },
    dm: { bg: '#ff6b9d', label: 'DM' },
    hint: { bg: '#ffd166', label: 'Hint' },
    vote: { bg: '#a855f7', label: 'Vote' },
    results: { bg: '#2ed573', label: 'Results' },
  };
  const s = styles[type] ?? { bg: '#888', label: type };
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: '0.7rem',
        fontWeight: 700,
        background: s.bg,
        color: '#000',
        marginRight: 6,
        textTransform: 'uppercase',
      }}
    >
      {s.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Activity item (All Activity tab)
// ---------------------------------------------------------------------------
function ActivityItem({ entry, players }) {
  const name = (id) => getPlayerName(players, id);

  switch (entry.type) {
    case 'group':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="group" />
          <strong>{name(entry.sender)}</strong>: {entry.content}
        </div>
      );
    case 'dm':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="dm" />
          <strong>{name(entry.sender)}</strong> &rarr; <strong>{name(entry.recipient)}</strong>: {entry.content}
        </div>
      );
    case 'hint':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="hint" />
          <strong>{name(entry.sender)}</strong>: &ldquo;{entry.content}&rdquo;
        </div>
      );
    case 'vote':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="vote" />
          <strong>{name(entry.sender)}</strong> &rarr; <strong>{name(entry.content)}</strong>
        </div>
      );
    case 'results':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="results" />
          Round results in
        </div>
      );
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Chat message (Group / DM tabs)
// ---------------------------------------------------------------------------
function ChatMessage({ msg, players }) {
  const name = getPlayerName(players, msg.sender);
  return (
    <div className="host-chat-message">
      <span className="host-chat-sender">{name}</span>
      <span className="host-chat-text">{msg.content}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Lobby Setup Panel
// ---------------------------------------------------------------------------
function LobbySetup({ playerList, connectedCount }) {
  const [availableAgents, setAvailableAgents] = useState([]);
  const [addingAgent, setAddingAgent] = useState(null);
  // counts how many times each agent type has been added
  const [addedCounts, setAddedCounts] = useState({});
  const [gameStarted, setGameStarted] = useState(false);
  const [startLoading, setStartLoading] = useState(false);

  // Fetch available agent types
  useEffect(() => {
    fetch('/agents')
      .then(res => res.json())
      .then(data => {
        const seen = new Set();
        const unique = (data.agents ?? []).filter(a => {
          if (seen.has(a.name)) return false;
          seen.add(a.name);
          return true;
        });
        setAvailableAgents(unique);
      })
      .catch((e) => console.error('Failed to fetch agents:', e));
  }, []);

  const handleAddAgent = useCallback(async (agentType) => {
    setAddingAgent(agentType);
    try {
      await fetch('/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agents: [{ type: agentType }] }),
      });
      setAddedCounts(prev => ({ ...prev, [agentType]: (prev[agentType] ?? 0) + 1 }));
    } catch (e) {
      console.error('Failed to add agent:', e);
    }
    setAddingAgent(null);
  }, []);

  const handleRemoveAgent = useCallback(async (agentType) => {
    const count = addedCounts[agentType] ?? 0;
    if (count === 0) return;
    // Ask server to remove one instance of this agent type from pending
    try {
      await fetch('/setup/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: agentType }),
      });
      setAddedCounts(prev => ({ ...prev, [agentType]: Math.max(0, (prev[agentType] ?? 0) - 1) }));
    } catch (e) {
      console.error('Failed to remove agent:', e);
    }
  }, [addedCounts]);

  const handleStartGame = useCallback(async () => {
    setStartLoading(true);
    try {
      await fetch('/start', { method: 'POST' });
      setGameStarted(true);
    } catch (e) {
      console.error('Failed to start game:', e);
    }
    setStartLoading(false);
  }, []);

  const totalAgentsQueued = Object.values(addedCounts).reduce((a, b) => a + b, 0);

  return (
    <div className="host-lobby">
      <h2>Game Setup</h2>

      <div className="host-setup-panel">
        {/* Left: Connected Players */}
        <div className="host-setup-section">
          <h3>Connected Players</h3>
          <ul className="host-player-list">
            {playerList.map(p => (
              <li key={p.player_id} className="host-player-item">
                <span className={`indicator ${p.connected ? 'online' : 'offline'}`} />
                <span>{p.display_name}</span>
              </li>
            ))}
          </ul>
          <p style={{ color: 'var(--text-secondary)', margin: '12px 0', fontSize: '0.85rem' }}>
            {connectedCount} human(s) + {totalAgentsQueued} agent(s) queued
          </p>
        </div>

        {/* Right: Available Agents */}
        <div className="host-setup-section">
          <h3>Add AI Agents</h3>

          <div className="host-agent-grid">
            {availableAgents.map(agent => {
              const count = addedCounts[agent.type] ?? 0;
              const isAdding = addingAgent === agent.type;
              return (
                <div key={agent.type} className="host-agent-slot" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
                  <span className="host-agent-slot-label">{agent.name}</span>

                  {count > 0 ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <button
                        onClick={() => handleRemoveAgent(agent.type)}
                        disabled={gameStarted}
                        style={{
                          background: 'transparent',
                          border: '1px solid var(--border)',
                          color: 'var(--text-secondary)',
                          borderRadius: '4px',
                          width: '26px', height: '26px',
                          cursor: 'pointer', fontSize: '1rem', lineHeight: 1,
                        }}
                      >−</button>
                      <span style={{
                        minWidth: '22px', textAlign: 'center',
                        fontWeight: 700, color: 'var(--accent)',
                      }}>{count}</span>
                      <button
                        onClick={() => handleAddAgent(agent.type)}
                        disabled={isAdding || gameStarted}
                        style={{
                          background: 'transparent',
                          border: '1px solid var(--border)',
                          color: 'var(--text-secondary)',
                          borderRadius: '4px',
                          width: '26px', height: '26px',
                          cursor: 'pointer', fontSize: '1rem', lineHeight: 1,
                        }}
                      >+</button>
                      <span style={{
                        color: '#2ed573', fontSize: '0.78rem',
                        fontWeight: 600, minWidth: '52px',
                      }}>✓ Added</span>
                    </div>
                  ) : (
                    <button
                      className="btn-host-action"
                      onClick={() => handleAddAgent(agent.type)}
                      disabled={!agent.available || isAdding || gameStarted}
                      style={{ minWidth: '110px' }}
                    >
                      {isAdding ? 'Adding...' : agent.available ? 'Add to Game' : 'Unavailable'}
                    </button>
                  )}
                </div>
              );
            })}
            {availableAgents.length === 0 && (
              <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Loading agents...</p>
            )}
          </div>

          {/* Start Game */}
          <div className="host-setup-actions" style={{ marginTop: '16px' }}>
            <button
              className="btn-start"
              onClick={handleStartGame}
              disabled={startLoading || gameStarted || connectedCount < 1}
            >
              {gameStarted ? 'Game Started!' : 'Start Game'}
            </button>
            {connectedCount < 1 && !gameStarted && (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '6px' }}>
                Waiting for at least 1 human player...
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Communication Tabs (center panel during game)
// ---------------------------------------------------------------------------
function CommTabs({ allActivity, players, activeTab, activeDMPair, dispatch }) {
  // Derive DM conversations from allActivity (no state duplication)
  const dmConversations = useMemo(() => {
    const convos = {};
    allActivity.filter(e => e.type === 'dm').forEach(dm => {
      const key = [dm.sender, dm.recipient].sort().join('_');
      if (!convos[key]) convos[key] = [];
      convos[key].push(dm);
    });
    return convos;
  }, [allActivity]);

  const dmPairs = useMemo(() => {
    return Object.entries(dmConversations).map(([key, msgs]) => ({
      key,
      players: key.split('_'),
      count: msgs.length,
    }));
  }, [dmConversations]);

  // Filtered messages for Group tab
  const groupMessages = useMemo(() => {
    return allActivity.filter(e => e.type === 'group');
  }, [allActivity]);

  const setTab = useCallback((tab) => {
    dispatch({ type: UI.HOST_SET_COMM_TAB, data: tab });
  }, [dispatch]);

  const setDMPair = useCallback((pair) => {
    dispatch({ type: UI.HOST_SET_DM_PAIR, data: pair });
  }, [dispatch]);

  return (
    <div className="host-comm-panel">
      {/* Tab bar */}
      <div className="host-comm-tabs">
        <button
          className={`host-comm-tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setTab('all')}
        >
          All
        </button>
        <button
          className={`host-comm-tab ${activeTab === 'group' ? 'active' : ''}`}
          onClick={() => setTab('group')}
        >
          Group
        </button>
        <button
          className={`host-comm-tab ${activeTab === 'dms' ? 'active' : ''}`}
          onClick={() => setTab('dms')}
        >
          DMs
        </button>
      </div>

      {/* Tab content */}
      <div className="host-comm-content">
        {activeTab === 'all' && (
          <div className="host-activity-feed">
            {allActivity.length === 0 && (
              <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                No activity yet...
              </p>
            )}
            {allActivity.map((entry, i) => (
              <ActivityItem key={i} entry={entry} players={players} />
            ))}
          </div>
        )}

        {activeTab === 'group' && (
          <div className="host-activity-feed">
            {groupMessages.length === 0 && (
              <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                No group messages yet...
              </p>
            )}
            {groupMessages.map((msg, i) => (
              <ChatMessage key={i} msg={msg} players={players} />
            ))}
          </div>
        )}

        {activeTab === 'dms' && (
          <div className="host-dm-panel">
            {/* DM pair selector */}
            <div className="host-dm-pairs">
              {dmPairs.length === 0 && (
                <p style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.85rem' }}>
                  No DMs yet...
                </p>
              )}
              {dmPairs.map(pair => (
                <button
                  key={pair.key}
                  className={`host-dm-pair-item ${activeDMPair === pair.key ? 'active' : ''}`}
                  onClick={() => setDMPair(pair.key)}
                >
                  <span>
                    {getPlayerName(players, pair.players[0])} &harr; {getPlayerName(players, pair.players[1])}
                  </span>
                  <span className="host-dm-badge">{pair.count}</span>
                </button>
              ))}
            </div>

            {/* DM conversation view */}
            {activeDMPair && dmConversations[activeDMPair] && (
              <div className="host-activity-feed">
                {dmConversations[activeDMPair].map((msg, i) => (
                  <ChatMessage key={i} msg={msg} players={players} />
                ))}
              </div>
            )}
            {activeDMPair && !dmConversations[activeDMPair] && (
              <p style={{ color: 'var(--text-muted)', fontStyle: 'italic', padding: '10px' }}>
                No messages in this conversation.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main HostDashboard
// ---------------------------------------------------------------------------
export default function HostDashboard({ state, dispatch, send }) {
  const {
    phase, currentRound, topic, secretWord, imposterId,
    players, allActivity, hintOrder, hints,
    cumulativeScores, timerSeconds, timerActive,
    revealData, finalScores, numRounds,
    hostActiveCommTab, hostActiveDMPair,
  } = state;

  const playerList = useMemo(() => Object.values(players), [players]);
  const phaseLabel = PHASE_LABELS[phase] ?? phase;

  // Lobby view — host controls game setup
  if (!phase || phase === 'LOBBY') {
    const connectedCount = playerList.filter(p => p.connected).length;
    return (
      <div className="screen host-dashboard">
        <div className="host-header">
          <h1>THE SANDBOX &mdash; HOST VIEW</h1>
          <span className="host-badge">HOST</span>
        </div>
        <LobbySetup playerList={playerList} connectedCount={connectedCount} />
      </div>
    );
  }

  // Game view — three-column layout with tabbed comms
  return (
    <div className="screen host-dashboard">
      {/* Header bar */}
      <div className="host-header">
        <h1>THE SANDBOX &mdash; HOST VIEW</h1>
        <div className="host-header-info">
          {currentRound > 0 && <span className="host-round">Round {currentRound}/{numRounds}</span>}
          <span className="host-phase">{phaseLabel}</span>
          {timerActive && timerSeconds > 0 && (
            <span className="host-timer">{formatTime(timerSeconds)}</span>
          )}
        </div>
      </div>

      <div className="host-body">
        {/* Left column — Player list */}
        <div className="host-panel host-players-panel">
          <h3>Players</h3>
          <ul className="host-player-list">
            {playerList.map(p => {
              const isImposter = p.player_id === imposterId;
              return (
                <li
                  key={p.player_id}
                  className={`host-player-item ${isImposter ? 'host-imposter' : ''}`}
                >
                  {isImposter && <span className="host-imposter-flag" title="Imposter">&#9873;</span>}
                  <span className={`indicator ${p.connected ? 'online' : 'offline'}`} />
                  <span>{p.display_name}</span>
                  {isImposter && <span className="host-imposter-label">IMPOSTER</span>}
                </li>
              );
            })}
          </ul>

          {/* Hints summary when available */}
          {Object.keys(hints).length > 0 && (
            <div className="host-hints-summary">
              <h4>Hints</h4>
              {hintOrder.map(pid => (
                <div key={pid} className="host-hint-entry">
                  <strong>{getPlayerName(players, pid)}:</strong>{' '}
                  {hints[pid] ? `"${hints[pid]}"` : '...'}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Center column — Tabbed communication panel */}
        <div className="host-panel host-activity-panel">
          <CommTabs
            allActivity={allActivity}
            players={players}
            activeTab={hostActiveCommTab}
            activeDMPair={hostActiveDMPair}
            dispatch={dispatch}
          />
        </div>

        {/* Right column — Round info + scores */}
        <div className="host-panel host-info-panel">
          <h3>Round Info</h3>
          {topic && (
            <div className="host-info-block">
              <div className="host-info-row">
                <span className="host-info-label">Topic</span>
                <span className="host-info-value">{topic}</span>
              </div>
              <div className="host-info-row">
                <span className="host-info-label">Secret Word</span>
                <span className="host-info-value host-secret">{secretWord ?? '???'}</span>
              </div>
              <div className="host-info-row">
                <span className="host-info-label">Imposter</span>
                <span className="host-info-value" style={{ color: 'var(--imposter-color)' }}>
                  {imposterId ? getPlayerName(players, imposterId) : '—'}
                </span>
              </div>
            </div>
          )}

          {Object.keys(cumulativeScores).length > 0 && (
            <div className="host-scores-block">
              <h4>Scores</h4>
              {Object.entries(cumulativeScores)
                .sort(([, a], [, b]) => b - a)
                .map(([pid, score]) => (
                  <div key={pid} className="host-score-row">
                    <span>{getPlayerName(players, pid)}</span>
                    <span className="host-score-value">{score}</span>
                  </div>
                ))}
            </div>
          )}

          {/* Grand reveal data */}
          {revealData && (
            <div className="host-reveal-block">
              <h4>Final Reveal</h4>
              {revealData.leaderboard?.map(entry => (
                <div key={entry.player_id} className="host-score-row">
                  <span>#{entry.rank} {getPlayerName(players, entry.player_id)}</span>
                  <span className="host-score-value">{entry.score}</span>
                </div>
              ))}
            </div>
          )}

          {/* Game finished */}
          {phase === 'FINISHED' && finalScores && (
            <div className="host-finished-block">
              <h4>Game Over</h4>
              <p style={{ color: 'var(--text-secondary)' }}>
                Final scores recorded.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
