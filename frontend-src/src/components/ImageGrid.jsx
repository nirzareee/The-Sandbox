import { getCardEmoji, getCardColor } from '../utils';

export default function ImageGrid({ grid, role, secretWord, topic }) {
  return (
    <div className="image-grid">
      {grid.map((item, i) => {
        // Support both legacy string entries and {label, image_path} objects.
        const label = typeof item === 'string' ? item : item.label;
        const imagePath = typeof item === 'string' ? '' : (item.image_path || '');
        const isSecret =
          role === 'INSIDER' &&
          secretWord &&
          label.toLowerCase() === secretWord.toLowerCase();
        return (
          <div key={i} className={`grid-card${isSecret ? ' secret' : ''}`}>
            <div
              className="card-inner"
              style={imagePath ? undefined : { backgroundColor: getCardColor(label) }}
            >
              {imagePath ? (
                <img
                  src={imagePath}
                  alt={label}
                  className="card-image"
                  loading="lazy"
                  onError={(e) => {
                    // If image fails to load, swap to emoji fallback
                    e.currentTarget.style.display = 'none';
                    const parent = e.currentTarget.parentElement;
                    if (parent) {
                      parent.style.backgroundColor = getCardColor(label);
                      const span = document.createElement('span');
                      span.className = 'card-emoji';
                      span.textContent = getCardEmoji(label);
                      parent.appendChild(span);
                    }
                  }}
                />
              ) : (
                <span className="card-emoji">{getCardEmoji(label)}</span>
              )}
            </div>
            <span className="card-label">{label}</span>
          </div>
        );
      })}
    </div>
  );
}
