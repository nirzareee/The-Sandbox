import { SERVER, UI } from './actions.js';
import { initialState } from './initialState.js';

let toastCounter = 0;

function makeToast(message) {
  return { id: `toast-${++toastCounter}`, message };
}

export function gameReducer(state, action) {
  const { type, data } = action;

  switch (type) {
    // ── Server events ──────────────────────────────────────────────

    case SERVER.PLAYER_JOINED: {
      const p = data;
      const existing = state.players[p.player_id] ?? {};
      return {
        ...state,
        players: {
          ...state.players,
          [p.player_id]: {
            ...existing,
            player_id: p.player_id,
            display_name: p.display_name,
            connected: true,
            // Hosts receive player_type + model_name; players don't — only
            // overwrite if present so we never clobber existing host info.
            ...(p.player_type !== undefined ? { player_type: p.player_type } : {}),
            ...(p.model_name !== undefined ? { model_name: p.model_name } : {}),
            ...(p.real_name !== undefined ? { real_name: p.real_name } : {}),
          },
        },
      };
    }

    case SERVER.PLAYER_LEFT: {
      const pid = data.player_id;
      const existing = state.players[pid];
      if (!existing) return state;
      return {
        ...state,
        players: {
          ...state.players,
          [pid]: { ...existing, connected: false },
        },
      };
    }

    case SERVER.PLAYER_ROSTER: {
      const roster = data.roster || {};
      const updatedPlayers = { ...state.players };
      for (const [pid, displayName] of Object.entries(roster)) {
        if (updatedPlayers[pid]) {
          updatedPlayers[pid] = { ...updatedPlayers[pid], display_name: displayName };
        } else {
          updatedPlayers[pid] = { player_id: pid, display_name: displayName, connected: true };
        }
      }
      return { ...state, players: updatedPlayers };
    }

    case SERVER.HOST_ROSTER: {
      // Host-only event: overlays player_type + model_name on top of existing
      // player objects. Players never receive this event.
      const roster = data.roster || {};
      const updatedPlayers = { ...state.players };
      for (const [pid, info] of Object.entries(roster)) {
        updatedPlayers[pid] = {
          ...(updatedPlayers[pid] ?? { player_id: pid }),
          display_name: info.display_name,
          player_type: info.player_type,
          model_name: info.model_name,
          real_name: info.real_name,
          connected: info.connected,
        };
      }
      return { ...state, players: updatedPlayers };
    }

    case SERVER.GAME_CREATED:
      return { ...state, sessionId: data.session_id, numRounds: data.num_rounds || 5 };

    case SERVER.ROUND_START: {
      const isHostRound = data.role === 'HOST';
      // Prefer grid_items (objects with label+image_path); fall back to image_grid (strings)
      const gridItems = data.grid_items?.length
        ? data.grid_items
        : (data.image_grid ?? []).map((label) => ({ label, image_path: '' }));
      return {
        ...state,
        currentRound: data.round_number,
        role: data.role,
        topic: data.topic,
        secretWord: data.secret_word ?? null,
        grid: gridItems,
        hintOrder: data.hint_order ?? [],
        hints: {},
        votes: {},
        chatMessages: [],
        dmMessages: {},
        unreadDMs: {},
        activeTab: 'group',
        hasVoted: false,
        isMyTurn: false,
        currentHinter: null,
        hintSubmitted: false,
        hintsExpected: 0,
        allHintsIn: false,
        readingCountdown: 0,
        phase: 'ASSIGNMENT',
        // Host-specific state
        isHost: isHostRound ? true : state.isHost,
        imposterId: isHostRound ? data.imposter_id : null,
        allActivity: [],
      };
    }

    case SERVER.HINT_PHASE_START:
      return {
        ...state,
        phase: 'HINT',
        hintOrder: data.hint_order ?? state.hintOrder,
        timerSeconds: data.total_duration ?? 0,
        timerActive: true,
        hintsExpected: data.expected_hints ?? 0,
        hintSubmitted: false,
        allHintsIn: false,
        readingCountdown: 0,
        hintMaxChars: data.max_chars ?? state.hintMaxChars,
        hintMaxWords: data.max_words ?? state.hintMaxWords,
      };

    case SERVER.YOUR_TURN:
      return { ...state, isMyTurn: true, currentHinter: state.playerId };

    case SERVER.HINT_SUBMITTED: {
      const isMe = data.player_id === state.playerId;
      const newState = {
        ...state,
        hints: { ...state.hints, [data.player_id]: data.hint },
        hintSubmitted: isMe ? true : state.hintSubmitted,
      };
      if (state.isHost) {
        newState.allActivity = [...state.allActivity, {
          type: 'hint',
          sender: data.player_id,
          content: data.hint,
          timestamp: Date.now(),
        }];
      }
      return newState;
    }

    case SERVER.ALL_HINTS_IN:
      return {
        ...state,
        allHintsIn: true,
        readingCountdown: data.reading_period ?? 10,
        timerSeconds: data.reading_period ?? 10,
        timerActive: true,
      };

    case SERVER.HINT_PHASE_END:
      return { ...state, isMyTurn: false, timerActive: false };

    case SERVER.DISCUSSION_START:
      return {
        ...state,
        phase: 'DISCUSSION',
        timerSeconds: data.duration ?? 0,
        timerActive: true,
      };

    case SERVER.GROUP_MESSAGE: {
      // Server sends player_id, not sender, for group_message
      const isOwnMsg = data.player_id === state.playerId;
      const msg = {
        sender: data.player_id,
        content: data.content,
        type: 'group',
        timestamp: Date.now(),
      };
      // Hosts see all messages in the activity feed
      if (state.isHost) {
        return {
          ...state,
          chatMessages: [...state.chatMessages, msg],
          allActivity: [...state.allActivity, msg],
        };
      }
      if (isOwnMsg) return state;
      return {
        ...state,
        chatMessages: [...state.chatMessages, msg],
      };
    }

    case SERVER.DM: {
      const dmMsg = {
        sender: data.sender,
        recipient: data.recipient,
        content: data.content,
        timestamp: data.timestamp ?? Date.now(),
      };
      // Hosts see all DMs in the activity feed
      if (state.isHost) {
        const activityEntry = { ...dmMsg, type: 'dm' };
        return {
          ...state,
          allActivity: [...state.allActivity, activityEntry],
        };
      }
      if (data.sender === state.playerId) return state;
      const otherPlayer = data.sender;
      const existing = state.dmMessages[otherPlayer] ?? [];
      const isActiveTab = state.activeTab === `dm-${otherPlayer}`;
      return {
        ...state,
        dmMessages: {
          ...state.dmMessages,
          [otherPlayer]: [...existing, dmMsg],
        },
        unreadDMs: isActiveTab
          ? state.unreadDMs
          : {
              ...state.unreadDMs,
              [otherPlayer]: (state.unreadDMs[otherPlayer] ?? 0) + 1,
            },
      };
    }

    case SERVER.DISCUSSION_END:
      return { ...state, timerActive: false };

    case SERVER.VOTING_START:
      return {
        ...state,
        phase: 'VOTING',
        candidates: data.candidates ?? [],
        hasVoted: false,
        timerSeconds: data.duration ?? 0,
        timerActive: true,
      };

    case SERVER.VOTING_END:
      return { ...state, timerActive: false };

    case SERVER.ROUND_RESULTS: {
      const resultsState = {
        ...state,
        phase: 'RESULTS',
        votes: data.votes ?? {},
        imposterId: data.imposter_id ?? state.imposterId,
        imposterCaught: data.imposter_caught ?? false,
        correctVoters: data.correct_voters ?? [],
        roundScores: data.scores_this_round ?? {},
        cumulativeScores: data.cumulative_scores ?? {},
        timerActive: false,
      };
      if (state.isHost) {
        // Add vote entries to activity feed
        const voteEntries = Object.entries(data.votes ?? {}).map(([voter, target]) => ({
          type: 'vote',
          sender: voter,
          content: target,
          timestamp: Date.now(),
        }));
        resultsState.allActivity = [...state.allActivity, ...voteEntries, {
          type: 'results',
          content: data,
          timestamp: Date.now(),
        }];
      }
      return resultsState;
    }

    case SERVER.REFLECTION_START:
      return {
        ...state,
        phase: 'REFLECTION',
        timerSeconds: data.duration ?? 0,
        timerActive: true,
      };

    case SERVER.REFLECTION_END:
      return { ...state, timerActive: false };

    case SERVER.TIME_WARNING:
      return { ...state, timerSeconds: data.seconds_remaining };

    case SERVER.META_GAME_START:
      return {
        ...state,
        phase: 'META_GAME',
        metaGamePlayers: data.players ?? {},
        timerSeconds: data.duration ?? 0,
        timerActive: true,
        classificationSubmitted: false,
      };

    case SERVER.META_GAME_END:
      return { ...state, timerActive: false };

    case SERVER.SURVEY_START:
      return {
        ...state,
        phase: 'SURVEY',
        timerSeconds: data.duration ?? 0,
        timerActive: true,
        surveySubmitted: false,
      };

    case SERVER.SURVEY_END:
      return { ...state, timerActive: false };

    case SERVER.GRAND_REVEAL:
      return { ...state, phase: 'REVEAL', revealData: data };

    case SERVER.GAME_FINISHED:
      return { ...state, phase: 'FINISHED', finalScores: data };

    case SERVER.ERROR:
      return {
        ...state,
        toasts: [...state.toasts, makeToast(data.message ?? 'Unknown error')],
      };

    // ── UI actions ─────────────────────────────────────────────────

    case UI.SET_CONNECTED:
      return { ...state, connected: data };

    case UI.SET_PLAYER:
      return {
        ...state,
        playerId: data.playerId,
        playerName: data.playerName,
        isHost: data.isHost ?? state.isHost,
      };

    case UI.SET_ACTIVE_TAB: {
      const tab = data;
      // Clear unread count if switching to a DM tab
      if (tab.startsWith('dm-')) {
        const dmPlayerId = tab.slice(3);
        const { [dmPlayerId]: _, ...rest } = state.unreadDMs;
        return { ...state, activeTab: tab, unreadDMs: rest };
      }
      return { ...state, activeTab: tab };
    }

    case UI.ADD_LOCAL_MESSAGE: {
      const msg = {
        sender: state.playerId,
        content: data.content,
        type: data.type ?? 'chat',
        timestamp: Date.now(),
      };
      return { ...state, chatMessages: [...state.chatMessages, msg] };
    }

    case UI.ADD_LOCAL_DM: {
      const target = data.target;
      const dmMsg = {
        sender: state.playerId,
        content: data.content,
        timestamp: Date.now(),
      };
      const existing = state.dmMessages[target] ?? [];
      return {
        ...state,
        dmMessages: {
          ...state.dmMessages,
          [target]: [...existing, dmMsg],
        },
      };
    }

    case UI.MARK_DM_READ: {
      const { [data]: _, ...rest } = state.unreadDMs;
      return { ...state, unreadDMs: rest };
    }

    case UI.SET_VOTED:
      return { ...state, hasVoted: true };

    case UI.TICK_TIMER: {
      const next = state.timerSeconds - 1;
      if (next <= 0) return { ...state, timerSeconds: 0, timerActive: false };
      return { ...state, timerSeconds: next };
    }

    case UI.SET_TIMER:
      return {
        ...state,
        timerSeconds: data.seconds,
        timerActive: data.active,
      };

    case UI.ADD_TOAST:
      return {
        ...state,
        toasts: [...state.toasts, makeToast(data)],
      };

    case UI.REMOVE_TOAST:
      return {
        ...state,
        toasts: state.toasts.filter((t) => t.id !== data),
      };

    case UI.HOST_SET_COMM_TAB:
      return { ...state, hostActiveCommTab: data, hostActiveDMPair: null };

    case UI.HOST_SET_DM_PAIR:
      return { ...state, hostActiveDMPair: data };

    default:
      return state;
  }
}

export default gameReducer;
