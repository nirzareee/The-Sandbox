export function randomChars(n) {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < n; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

export function formatTime(seconds) {
  if (seconds == null || seconds < 0) return '--:--';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

export function getPlayerName(players, playerId) {
  return players[playerId]?.display_name || playerId;
}

const AVATAR_COLORS = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#e84393', '#00b894', '#6c5ce7',
];

export function getAvatarColor(playerId) {
  let hash = 0;
  const str = String(playerId);
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

export function getInitials(name) {
  if (!name) return '?';
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

const EMOJI_MAP = {
  // Fruits
  Mango: '🥭', Apple: '🍎', Banana: '🍌', Strawberry: '🍓',
  Watermelon: '🍉', Grape: '🍇', Pineapple: '🍍', Cherry: '🍒',
  Orange: '🍊', Kiwi: '🥝',
  // Movies
  Inception: '🌀', Titanic: '🚢', Avatar: '👽', 'The Matrix': '💊',
  'Forrest Gump': '🏃', 'The Godfather': '🎩', 'Jurassic Park': '🦖',
  Gladiator: '⚔️', Interstellar: '🚀', Jaws: '🦈',
  // Animals
  Penguin: '🐧', Lion: '🦁', Elephant: '🐘', Dolphin: '🐬',
  Eagle: '🦅', Tiger: '🐯', Giraffe: '🦒', Panda: '🐼',
  Wolf: '🐺', Octopus: '🐙',
  // Sports
  Tennis: '🎾', Soccer: '⚽', Basketball: '🏀', Swimming: '🏊',
  Boxing: '🥊', Cricket: '🏏', Golf: '⛳', Volleyball: '🏐',
  Rugby: '🏉', Archery: '🏹',
  // Musical Instruments
  Drums: '🥁', Piano: '🎹', Guitar: '🎸', Violin: '🎻',
  Trumpet: '🎺', Flute: '🪈', Saxophone: '🎷', Cello: '🎻',
  Harp: '🪕', Accordion: '🪗',
  // Vehicles
  Submarine: '🚢', Helicopter: '🚁', Motorcycle: '🏍️', Sailboat: '⛵',
  Bicycle: '🚲', Train: '🚂', Ambulance: '🚑', Tractor: '🚜',
  Canoe: '🛶', Hovercraft: '🛥️',
  // Desserts
  Tiramisu: '🍰', Cheesecake: '🧁', Brownie: '🍫', Macaron: '🧁',
  Baklava: '🥮', Churros: '🍩', Gelato: '🍨', Pavlova: '🎂',
  Mochi: '🍡', Cannoli: '🥐',
};

export function getCardEmoji(label) {
  return EMOJI_MAP[label] || '❓';
}

const CARD_COLORS = [
  '#ff9ff3', '#feca57', '#54a0ff', '#5f27cd', '#01a3a4',
  '#ee5a24', '#c8d6e5', '#ff6b6b', '#48dbfb', '#1dd1a1',
];

export function getCardColor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = label.charCodeAt(i) + ((hash << 5) - hash);
  return CARD_COLORS[Math.abs(hash) % CARD_COLORS.length];
}
