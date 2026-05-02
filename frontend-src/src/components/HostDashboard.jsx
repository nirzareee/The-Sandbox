import { useMemo, useState, useEffect, useCallback } from 'react';
import { UI } from '../state/actions.js';
import { getAvatarColor, getInitials } from '../utils.js';

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

function formatClock(ts) {
  if (!ts) return '';
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return '';
  }
}

function truncate(text, n = 48) {
  if (!text) return '';
  return text.length > n ? text.slice(0, n - 1) + '…' : text;
}

// ---------------------------------------------------------------------------
// Avatar component
// ---------------------------------------------------------------------------
function Avatar({ playerId, players, size = 'md' }) {
  const name = getPlayerName(players, playerId);
  return (
    <div
      className={`host-avatar host-avatar-${size}`}
      style={{ backgroundColor: getAvatarColor(playerId) }}
      title={name}
    >
      {getInitials(name)}
    </div>
  );
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
    <span className="host-activity-badge" style={{ background: s.bg }}>
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
          <Avatar playerId={entry.sender} players={players} size="xs" />
          <span className="host-activity-text">
            <strong>{name(entry.sender)}</strong>: {entry.content}
          </span>
        </div>
      );
    case 'dm':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="dm" />
          <Avatar playerId={entry.sender} players={players} size="xs" />
          <span className="host-activity-text">
            <strong>{name(entry.sender)}</strong>
            <span className="host-arrow"> → </span>
            <strong>{name(entry.recipient)}</strong>: {entry.content}
          </span>
        </div>
      );
    case 'hint':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="hint" />
          <Avatar playerId={entry.sender} players={players} size="xs" />
          <span className="host-activity-text">
            <strong>{name(entry.sender)}</strong>: &ldquo;{entry.content}&rdquo;
          </span>
        </div>
      );
    case 'vote':
      return (
        <div className="host-activity-item">
          <ActivityBadge type="vote" />
          <Avatar playerId={entry.sender} players={players} size="xs" />
          <span className="host-activity-text">
            <strong>{name(entry.sender)}</strong>
            <span className="host-arrow"> → </span>
            <strong>{name(entry.content)}</strong>
          </span>
        </div>
      );
    case 'results':
      return (
        <div className="host-activity-item host-activity-results">
          <ActivityBadge type="results" />
          <span className="host-activity-text">Round results in</span>
        </div>
      );
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Host chat message (Group tab — single sender per line)
// ---------------------------------------------------------------------------
function HostGroupMessage({ msg, players }) {
  return (
    <div className="host-group-message">
      <Avatar playerId={msg.sender} players={players} size="sm" />
      <div className="host-group-body">
        <div className="host-group-meta">
          <span className="host-group-sender">{getPlayerName(players, msg.sender)}</span>
          {msg.timestamp && (
            <span className="host-group-time">{formatClock(msg.timestamp)}</span>
          )}
        </div>
        <div className="host-group-text">{msg.content}</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Host DM bubble — left/right aligned per sender so direction is obvious
// ---------------------------------------------------------------------------
function HostDMBubble({ msg, players, leftPlayerId }) {
  const isLeft = msg.sender === leftPlayerId;
  const senderName = getPlayerName(players, msg.sender);
  return (
    <div className={`host-dm-row ${isLeft ? 'from-left' : 'from-right'}`}>
      {isLeft && <Avatar playerId={msg.sender} players={players} size="sm" />}
      <div className="host-dm-bubble-wrap">
        <div className="host-dm-bubble-meta">
          <span className="host-dm-bubble-sender">{senderName}</span>
          {msg.timestamp && (
            <span className="host-dm-bubble-time">{formatClock(msg.timestamp)}</span>
          )}
        </div>
        <div className="host-dm-bubble">{msg.content}</div>
      </div>
      {!isLeft && <Avatar playerId={msg.sender} players={players} size="sm" />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Lobby Setup Panel
// ---------------------------------------------------------------------------
function LobbySetup({ playerList, connectedCount }) {
  const [availableAgents, setAvailableAgents] = useState([]);
  const [addingAgent, setAddingAgent] = useState(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [startLoading, setStartLoading] = useState(false);

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
    } catch (e) {
      console.error('Failed to add agent:', e);
    }
    setAddingAgent(null);
  }, []);

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

  return (
    <div className="host-lobby">
      <h2>Game Setup</h2>

      <div className="host-setup-panel">
        <div className="host-setup-section">
          <h3>Connected Players</h3>
          <ul className="host-player-list">
            {playerList.map(p => (
              <li key={p.player_id} className="host-player-item">
                <span className={`indicator ${p.connected ? 'online' : 'offline'}`} />
                <span className="host-player-item-name">{p.display_name}</span>
                <PlayerTypeBadge player={p} />
              </li>
            ))}
          </ul>
          <p className="host-muted-note">{connectedCount} player(s) connected</p>
        </div>

        <div className="host-setup-section">
          <h3>Available Agents</h3>
          <div className="host-agent-grid">
            {availableAgents.map(agent => (
              <div key={agent.type} className="host-agent-slot">
                <span className="host-agent-slot-label">{agent.name}</span>
                <button
                  className="btn-host-action"
                  onClick={() => handleAddAgent(agent.type)}
                  disabled={!agent.available || addingAgent !== null || gameStarted}
                >
                  {addingAgent === agent.type ? 'Adding...' : agent.available ? 'Add to Game' : 'Unavailable'}
                </button>
              </div>
            ))}
            {availableAgents.length === 0 && (
              <p className="host-muted-note host-italic">Loading agents...</p>
            )}
          </div>

          <div className="host-setup-actions">
            <button
              className="btn-start"
              onClick={handleStartGame}
              disabled={startLoading || gameStarted || connectedCount < 2}
            >
              {gameStarted ? 'Game Started!' : 'Start Game'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Communication Tabs (center panel during game)
// ---------------------------------------------------------------------------
function CommTabs({ allActivity, players, playerList, activeTab, activeDMPair, dispatch }) {
  // Derive DM conversations from allActivity (no state duplication)
  const dmConversations = useMemo(() => {
    const convos = {};
    allActivity.filter(e => e.type === 'dm').forEach(dm => {
      const key = [dm.sender, dm.recipient].sort().join('__');
      if (!convos[key]) convos[key] = [];
      convos[key].push(dm);
    });
    return convos;
  }, [allActivity]);

  // Generate ALL possible pairs from the player list (C(n,2)), not only pairs
  // that have exchanged DMs. This matches the user's spec: "if we have 3 people
  // then dm tab should have 3 tabs ( p1-p2, p1-p3, p2-p3)".
  const dmPairs = useMemo(() => {
    const sorted = [...playerList].sort((a, b) =>
      (a.display_name ?? '').localeCompare(b.display_name ?? '')
    );
    const pairs = [];
    for (let i = 0; i < sorted.length; i++) {
      for (let j = i + 1; j < sorted.length; j++) {
        const a = sorted[i].player_id;
        const b = sorted[j].player_id;
        const key = [a, b].sort().join('__');
        const msgs = dmConversations[key] ?? [];
        const last = msgs[msgs.length - 1];
        pairs.push({
          key,
          playerA: a,
          playerB: b,
          count: msgs.length,
          lastMsg: last,
          lastTs: last?.timestamp ?? 0,
        });
      }
    }
    return pairs;
  }, [dmConversations, playerList]);

  // Group chat messages (Group tab)
  const groupMessages = useMemo(() => {
    return allActivity.filter(e => e.type === 'group');
  }, [allActivity]);

  const setTab = useCallback((tab) => {
    dispatch({ type: UI.HOST_SET_COMM_TAB, data: tab });
  }, [dispatch]);

  const setDMPair = useCallback((pair) => {
    dispatch({ type: UI.HOST_SET_DM_PAIR, data: pair });
  }, [dispatch]);

  // Pick a default pair: most recent activity, otherwise first pair.
  const defaultPairKey = useMemo(() => {
    if (dmPairs.length === 0) return null;
    const withActivity = dmPairs.filter(p => p.lastTs > 0);
    if (withActivity.length > 0) {
      return withActivity.sort((a, b) => b.lastTs - a.lastTs)[0].key;
    }
    return dmPairs[0].key;
  }, [dmPairs]);

  // Auto-select default pair when switching into DMs tab or when pair disappears.
  useEffect(() => {
    if (activeTab !== 'dms' || !defaultPairKey) return;
    const currentStillValid = activeDMPair && dmPairs.some(p => p.key === activeDMPair);
    if (!currentStillValid) {
      setDMPair(defaultPairKey);
    }
  }, [activeTab, activeDMPair, dmPairs, defaultPairKey, setDMPair]);

  const activePair = activeDMPair
    ? dmPairs.find(p => p.key === activeDMPair)
    : null;
  const activeMessages = activePair ? (dmConversations[activePair.key] ?? []) : [];

  return (
    <div className="host-comm-panel">
      {/* Tab bar */}
      <div className="host-comm-tabs">
        <button
          className={`host-comm-tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setTab('all')}
        >
          All Activity
        </button>
        <button
          className={`host-comm-tab ${activeTab === 'group' ? 'active' : ''}`}
          onClick={() => setTab('group')}
        >
          Group Chat
          {groupMessages.length > 0 && (
            <span className="host-tab-count">{groupMessages.length}</span>
          )}
        </button>
        <button
          className={`host-comm-tab ${activeTab === 'dms' ? 'active' : ''}`}
          onClick={() => setTab('dms')}
        >
          Direct Messages
          {dmPairs.length > 0 && (
            <span className="host-tab-count">{dmPairs.length}</span>
          )}
        </button>
      </div>

      {/* Tab content */}
      <div className="host-comm-content">
        {activeTab === 'all' && (
          <div className="host-activity-feed">
            {allActivity.length === 0 && (
              <p className="host-empty-state">No activity yet...</p>
            )}
            {allActivity.map((entry, i) => (
              <ActivityItem key={i} entry={entry} players={players} />
            ))}
          </div>
        )}

        {activeTab === 'group' && (
          <div className="host-group-feed">
            {groupMessages.length === 0 && (
              <p className="host-empty-state">No group messages yet...</p>
            )}
            {groupMessages.map((msg, i) => (
              <HostGroupMessage key={i} msg={msg} players={players} />
            ))}
          </div>
        )}

        {activeTab === 'dms' && (
          <div className="host-dm-panel">
            {/* Horizontal sub-tab row: one tab per pair (all C(n,2) pairs) */}
            <div className="host-dm-subtabs" role="tablist">
              {dmPairs.length === 0 && (
                <span className="host-dm-subtabs-empty">
                  Waiting for players to join...
                </span>
              )}
              {dmPairs.map(pair => {
                const nameA = getPlayerName(players, pair.playerA);
                const nameB = getPlayerName(players, pair.playerB);
                const isActive = activeDMPair === pair.key;
                return (
                  <button
                    key={pair.key}
                    role="tab"
                    aria-selected={isActive}
                    className={`host-dm-subtab${isActive ? ' active' : ''}${pair.count === 0 ? ' is-empty' : ''}`}
                    onClick={() => setDMPair(pair.key)}
                    title={`${nameA} ↔ ${nameB} — ${pair.count} message${pair.count === 1 ? '' : 's'}`}
                  >
                    <Avatar playerId={pair.playerA} players={players} size="xs" />
                    <span className="host-dm-subtab-names">
                      <span>{nameA}</span>
                      <span className="host-dm-subtab-sep">–</span>
                      <span>{nameB}</span>
                    </span>
                    <Avatar playerId={pair.playerB} players={players} size="xs" />
                    {pair.count > 0 && (
                      <span className="host-dm-subtab-count">{pair.count}</span>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Conversation panel for the selected pair */}
            <div className="host-dm-convo">
              {!activePair ? (
                <p className="host-empty-state">
                  No player pairs available.
                </p>
              ) : (
                <>
                  <div className="host-dm-convo-header">
                    <div className="host-dm-convo-person host-dm-convo-left">
                      <Avatar playerId={activePair.playerA} players={players} size="md" />
                      <span className="host-dm-convo-name">
                        {getPlayerName(players, activePair.playerA)}
                      </span>
                    </div>
                    <span className="host-dm-convo-arrow">↔</span>
                    <div className="host-dm-convo-person host-dm-convo-right">
                      <span className="host-dm-convo-name">
                        {getPlayerName(players, activePair.playerB)}
                      </span>
                      <Avatar playerId={activePair.playerB} players={players} size="md" />
                    </div>
                    <span className="host-dm-convo-count">
                      {activeMessages.length} message{activeMessages.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  <div className="host-dm-convo-body">
                    {activeMessages.length === 0 ? (
                      <p className="host-empty-state host-italic">
                        No messages between these two yet.
                      </p>
                    ) : (
                      activeMessages.map((msg, i) => (
                        <HostDMBubble
                          key={i}
                          msg={msg}
                          players={players}
                          leftPlayerId={activePair.playerA}
                        />
                      ))
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Player-type badge — HUMAN / AI (with model name)
// ---------------------------------------------------------------------------
function PlayerTypeBadge({ player }) {
  if (!player) return null;
  const type = player.player_type;
  if (type === 'HUMAN') {
    const real = (player.real_name ?? '').trim();
    return (
      <span
        className="host-type-badge host-type-human"
        title={real ? `Human player — typed name: ${real}` : 'Human player'}
      >
        {real ? `HUMAN · ${real}` : 'HUMAN'}
      </span>
    );
  }
  if (type === 'AI') {
    const model = player.model_name ?? 'AI';
    return (
      <span
        className="host-type-badge host-type-ai"
        title={`AI agent — ${model}`}
      >
        AI · {model}
      </span>
    );
  }
  // Unknown type — render nothing (e.g., non-host clients won't have it)
  return null;
}

// ---------------------------------------------------------------------------
// Host player sidebar (with per-player stats + inline hint)
// ---------------------------------------------------------------------------
function HostPlayerSidebar({ playerList, imposterId, players, hints, hintOrder, allActivity }) {
  // Tally activity counts per player (cheap — allActivity is bounded by round length)
  const stats = useMemo(() => {
    const counts = {};
    for (const e of allActivity) {
      if (!counts[e.sender]) counts[e.sender] = { group: 0, dm: 0, vote: 0 };
      if (e.type === 'group') counts[e.sender].group += 1;
      else if (e.type === 'dm') counts[e.sender].dm += 1;
      else if (e.type === 'vote') counts[e.sender].vote += 1;
    }
    return counts;
  }, [allActivity]);

  return (
    <div className="host-players-panel host-panel">
      <h3>Players</h3>
      <ul className="host-player-list">
        {playerList.map(p => {
          const isImposter = p.player_id === imposterId;
          const s = stats[p.player_id] ?? { group: 0, dm: 0 };
          return (
            <li
              key={p.player_id}
              className={`host-player-card ${isImposter ? 'host-player-imposter' : ''}`}
            >
              <Avatar playerId={p.player_id} players={players} size="md" />
              <div className="host-player-card-body">
                <div className="host-player-card-top">
                  <span className="host-player-card-name">{p.display_name}</span>
                  {isImposter && <span className="host-imposter-tag">IMPOSTER</span>}
                </div>
                <div className="host-player-card-meta">
                  <PlayerTypeBadge player={p} />
                </div>
                <div className="host-player-card-stats">
                  <span className={`indicator ${p.connected ? 'online' : 'offline'}`} title={p.connected ? 'Online' : 'Offline'} />
                  <span title="Group messages">💬 {s.group}</span>
                  <span title="Direct messages">✉ {s.dm}</span>
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      {Object.keys(hints).length > 0 && (
        <div className="host-hints-summary">
          <h4>Hints</h4>
          {hintOrder.map(pid => (
            <div key={pid} className="host-hint-entry">
              <Avatar playerId={pid} players={players} size="xs" />
              <span className="host-hint-name">{getPlayerName(players, pid)}</span>
              <span className="host-hint-text">
                {hints[pid] ? `"${hints[pid]}"` : <span className="host-muted">…</span>}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Host right-side info panel (topic, secret, imposter, scores)
// ---------------------------------------------------------------------------
function HostInfoPanel({ topic, secretWord, imposterId, players, cumulativeScores, revealData, phase, finalScores }) {
  const sortedScores = Object.entries(cumulativeScores).sort(([, a], [, b]) => b - a);

  return (
    <div className="host-info-panel host-panel">
      <h3>Round Info</h3>
      {topic && (
        <>
          <div className="host-info-card host-info-topic">
            <span className="host-info-card-label">Topic</span>
            <span className="host-info-card-value">{topic}</span>
          </div>
          <div className="host-info-card host-info-secret">
            <span className="host-info-card-label">Secret Word</span>
            <span className="host-info-card-value">{secretWord ?? '???'}</span>
          </div>
          <div className="host-info-card host-info-imposter">
            <span className="host-info-card-label">Imposter</span>
            {imposterId ? (
              <div className="host-info-imposter-row">
                <Avatar playerId={imposterId} players={players} size="sm" />
                <span className="host-info-card-value">{getPlayerName(players, imposterId)}</span>
              </div>
            ) : (
              <span className="host-info-card-value host-muted">—</span>
            )}
          </div>
        </>
      )}

      {sortedScores.length > 0 && (
        <div className="host-scores-block">
          <h4>Scores</h4>
          {sortedScores.map(([pid, score], idx) => (
            <div key={pid} className="host-score-row">
              <span className="host-score-rank">#{idx + 1}</span>
              <Avatar playerId={pid} players={players} size="xs" />
              <span className="host-score-name">{getPlayerName(players, pid)}</span>
              <span className="host-score-value">{score}</span>
            </div>
          ))}
        </div>
      )}

      {revealData && (
        <div className="host-reveal-block">
          <h4>Final Reveal</h4>
          {revealData.leaderboard?.map(entry => (
            <div key={entry.player_id} className="host-score-row">
              <span className="host-score-rank">#{entry.rank}</span>
              <Avatar playerId={entry.player_id} players={players} size="xs" />
              <span className="host-score-name">{getPlayerName(players, entry.player_id)}</span>
              <span className="host-score-value">{entry.score}</span>
            </div>
          ))}
        </div>
      )}

      {phase === 'FINISHED' && finalScores && (
        <div className="host-finished-block">
          <h4>Game Over</h4>
          <p className="host-muted">Final scores recorded.</p>
        </div>
      )}
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

  // Game view — three-column layout
  return (
    <div className="screen host-dashboard">
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
        <HostPlayerSidebar
          playerList={playerList}
          imposterId={imposterId}
          players={players}
          hints={hints}
          hintOrder={hintOrder}
          allActivity={allActivity}
        />

        <div className="host-panel host-activity-panel">
          <CommTabs
            allActivity={allActivity}
            players={players}
            playerList={playerList}
            activeTab={hostActiveCommTab}
            activeDMPair={hostActiveDMPair}
            dispatch={dispatch}
          />
        </div>

        <HostInfoPanel
          topic={topic}
          secretWord={secretWord}
          imposterId={imposterId}
          players={players}
          cumulativeScores={cumulativeScores}
          revealData={revealData}
          phase={phase}
          finalScores={finalScores}
        />
      </div>
    </div>
  );
}
