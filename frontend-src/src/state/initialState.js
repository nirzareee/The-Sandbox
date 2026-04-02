export const initialState = {
  // Connection
  connected: false,
  playerId: null,
  playerName: null,

  // Game phase
  phase: 'LOBBY', // LOBBY | ASSIGNMENT | HINT | DISCUSSION | VOTING | RESULTS | REFLECTION | META_GAME | SURVEY | REVEAL | FINISHED
  sessionId: null,

  // Round state
  currentRound: 0,
  numRounds: 5,
  role: null,       // INSIDER | IMPOSTER
  topic: null,
  secretWord: null,  // null for imposter
  grid: [],          // flat array of label strings
  hintOrder: [],     // array of player_id strings
  currentHinter: null,
  isMyTurn: false,
  hints: {},         // { playerId: hintText }
  hintSubmitted: false,
  hintMaxChars: 60,
  hintMaxWords: 8,
  hintsExpected: 0,
  allHintsIn: false,
  readingCountdown: 0,

  // Players
  players: {},       // { playerId: { player_id, display_name, connected } }

  // Chat
  chatMessages: [],      // [{ sender, content, type, timestamp }]
  dmMessages: {},        // { playerId: [{ sender, content, timestamp }] }
  unreadDMs: {},         // { playerId: count }
  activeTab: 'group',   // 'group' | 'dm-{playerId}'

  // Voting
  hasVoted: false,
  candidates: [],
  votes: {},             // { voterId: targetId }

  // Scores
  roundScores: {},       // { playerId: points } (current round)
  cumulativeScores: {},  // { playerId: totalPoints }

  // Host mode
  isHost: false,
  allActivity: [],       // [{type, sender, recipient, content, timestamp}]
  hostActiveCommTab: 'all',   // 'all' | 'group' | 'dms'
  hostActiveDMPair: null,     // e.g. 'agent_1_agent_2' or null

  // Results
  imposterId: null,
  imposterCaught: false,
  correctVoters: [],

  // Timer
  timerSeconds: 0,
  timerActive: false,

  // Post-game
  metaGamePlayers: {},
  classificationSubmitted: false,
  surveySubmitted: false,
  revealData: null,      // { identities, leaderboard, classification_scores, confusion_matrix }
  finalScores: null,

  // UI
  toasts: [],            // [{ id, message }]
};
