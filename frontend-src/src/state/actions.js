// Server event types (25 total)
export const SERVER = {
  PLAYER_JOINED: 'player_joined',
  PLAYER_LEFT: 'player_left',
  PLAYER_ROSTER: 'player_roster',
  HOST_ROSTER: 'host_roster',
  GAME_CREATED: 'game_created',
  ROUND_START: 'round_start',
  HINT_PHASE_START: 'hint_phase_start',
  YOUR_TURN: 'your_turn',
  HINT_SUBMITTED: 'hint_submitted',
  ALL_HINTS_IN: 'all_hints_in',
  HINT_PHASE_END: 'hint_phase_end',
  DISCUSSION_START: 'discussion_start',
  GROUP_MESSAGE: 'group_message',
  DM: 'dm',
  DISCUSSION_END: 'discussion_end',
  VOTING_START: 'voting_start',
  VOTING_END: 'voting_end',
  ROUND_RESULTS: 'round_results',
  REFLECTION_START: 'reflection_start',
  REFLECTION_END: 'reflection_end',
  TIME_WARNING: 'time_warning',
  META_GAME_START: 'meta_game_start',
  META_GAME_END: 'meta_game_end',
  SURVEY_START: 'survey_start',
  SURVEY_END: 'survey_end',
  GRAND_REVEAL: 'grand_reveal',
  GAME_FINISHED: 'game_finished',
  ERROR: 'error',
};

// Client action types (6 total)
export const CLIENT = {
  GROUP_MESSAGE: 'GROUP_MESSAGE',
  DM: 'DM',
  HINT: 'HINT',
  VOTE: 'VOTE',
  CLASSIFICATION: 'CLASSIFICATION',
  SURVEY: 'SURVEY',
};

// UI-only actions
export const UI = {
  SET_CONNECTED: 'SET_CONNECTED',
  SET_PLAYER: 'SET_PLAYER',
  SET_ACTIVE_TAB: 'SET_ACTIVE_TAB',
  ADD_LOCAL_MESSAGE: 'ADD_LOCAL_MESSAGE',
  ADD_LOCAL_DM: 'ADD_LOCAL_DM',
  MARK_DM_READ: 'MARK_DM_READ',
  SET_VOTED: 'SET_VOTED',
  TICK_TIMER: 'TICK_TIMER',
  SET_TIMER: 'SET_TIMER',
  ADD_TOAST: 'ADD_TOAST',
  REMOVE_TOAST: 'REMOVE_TOAST',
  HOST_SET_COMM_TAB: 'HOST_SET_COMM_TAB',
  HOST_SET_DM_PAIR: 'HOST_SET_DM_PAIR',
};
