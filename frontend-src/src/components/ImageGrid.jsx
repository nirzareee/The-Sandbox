import { getCardEmoji, getCardColor } from '../utils';

export default function ImageGrid({ grid, role, secretWord, topic }) {
  return (
    <div className="image-grid">
      {grid.map((label, i) => {
        const isSecret =
          role === 'INSIDER' &&
          secretWord &&
          label.toLowerCase() === secretWord.toLowerCase();
        return (
          <div key={i} className={`grid-card${isSecret ? ' secret' : ''}`}>
            <div className="card-inner" style={{ backgroundColor: getCardColor(label) }}>
              <span className="card-emoji">{getCardEmoji(label)}</span>
            </div>
            <span className="card-label">{label}</span>
          </div>
        );
      })}
    </div>
  );
}
